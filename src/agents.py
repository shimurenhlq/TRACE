"""
AgenticSystem: Three-stage agent workflow for multi-modal document QA.
Implements Planner -> Navigator -> Reasoner pipeline.
"""

import re
import json
import torch
from typing import List, Dict, Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_core.models import UserMessage
from autogen_core import Image

from .config import Config
from .prompts import PLANNER_SYSTEM_PROMPT, NAVIGATOR_VLM_PROMPT_TEMPLATE, REASONER_SYSTEM_PROMPT


class AgenticSystem:
    """
    Three-stage agentic system for complex multi-modal document QA.

    Stages:
        1. Planner: Decomposes complex questions into sub-queries
        2. Navigator: Retrieves relevant pages using VLM + ColPali + MoLoRAG
        3. Reasoner: Synthesizes evidence and generates final answer
    """

    def __init__(self, config: Config):
        """
        Initialize the agentic system with model clients.

        Args:
            config: Configuration object containing model endpoints
        """
        self.config = config

        # Initialize model clients
        self.planner_client = OpenAIChatCompletionClient(
            api_key=config.planner.api_key,
            base_url=config.planner.base_url,
            model=config.planner.model,
            model_info=config.planner.model_info or {}
        )

        self.reasoner_client = OpenAIChatCompletionClient(
            api_key=config.reasoner.api_key,
            base_url=config.reasoner.base_url,
            model=config.reasoner.model,
            model_info=config.reasoner.model_info or {}
        )

    async def solve(self, question: str, options: List[str], book_env) -> Dict[str, Any]:
        """
        Solve a multi-modal QA question using the three-stage pipeline.

        Args:
            question: The question to answer
            options: List of answer options (e.g., ["A. ...", "B. ...", ...])
            book_env: BookEnvironment instance with loaded resources

        Returns:
            Dictionary containing:
                - pred_answer: Predicted answer (e.g., "A")
                - retrieved_indices: List of retrieved page indices
                - logs: Detailed execution logs
        """
        doc_logs = []

        # ==================== Stage 1: Planner ====================
        planner = AssistantAgent(
            name="planner",
            model_client=self.planner_client,
            system_message=PLANNER_SYSTEM_PROMPT
        )

        try:
            plan_res = await planner.on_messages(
                [TextMessage(content=f"Question: {question}", source="user")],
                cancellation_token=CancellationToken()
            )
            raw_plan = plan_res.chat_message.content
            doc_logs.append(f"[Planner] Raw Output: {raw_plan}")

            # Parse JSON list from planner output
            match = re.search(r"\[.*\]", raw_plan, re.DOTALL)
            if match:
                search_steps = json.loads(match.group(0))
            else:
                search_steps = [question]  # Fallback to original question
        except Exception as e:
            doc_logs.append(f"[Planner] Error: {e}")
            search_steps = [question]

        doc_logs.append(f"[Planner] Parsed Steps: {search_steps}")

        # ==================== Stage 2: Navigator ====================
        global_evidence_pool = []  # Stores evidence text
        all_retrieved_pages = set()  # Stores all relevant page indices
        global_score_accumulator = None  # Accumulates scores across steps

        for step_idx, step_query in enumerate(search_steps):
            doc_logs.append(f"\n--- Step {step_idx + 1}: {step_query} ---")

            # Calculate similarity scores for this query
            current_scores = book_env.retrieve_page_scores(step_query)
            if len(current_scores) > 0:
                if global_score_accumulator is None:
                    global_score_accumulator = torch.zeros_like(current_scores)
                global_score_accumulator += current_scores

            # Initialize search state
            step_accepted = []  # Pages accepted as relevant
            step_blacklist = set()  # Pages rejected as irrelevant
            search_stack = book_env.retrieve_initial_pages(step_query)  # Sorted ascending

            doc_logs.append(f"Initial Stack Size: {len(search_stack)}, Top: {[book_env.index_to_uid(i) for i in search_stack[-3:]]}")

            # Navigator loop: retrieve up to max_pages_per_step relevant pages
            while len(step_accepted) < self.config.max_pages_per_step and search_stack:
                curr_idx = search_stack.pop()  # Pop highest score
                curr_uid = book_env.index_to_uid(curr_idx)

                # Skip if already processed
                if curr_idx in step_blacklist or curr_idx in step_accepted:
                    continue

                img_path = book_env.get_image_path(curr_idx)
                if not img_path:
                    continue

                # VLM relevance filtering
                vlm_msg = UserMessage(
                    content=[
                        NAVIGATOR_VLM_PROMPT_TEMPLATE.format(query=step_query),
                        Image.from_file(img_path)
                    ],
                    source="user"
                )

                try:
                    resp = await book_env.vision_client.create([vlm_msg])
                    content = resp.content
                    doc_logs.append(f"[VLM {curr_uid}]: {content}")
                except Exception as e:
                    doc_logs.append(f"[VLM Error {curr_uid}]: {e}")
                    continue

                # Parse VLM verdict
                if "[IRRELEVANT]" in content:
                    step_blacklist.add(curr_idx)
                    doc_logs.append(f"[Navigator] {curr_uid} -> IRRELEVANT")
                else:
                    step_accepted.append(curr_idx)
                    all_retrieved_pages.add(curr_idx)
                    global_evidence_pool.append(f"Evidence from {curr_uid} for '{step_query}':\n{content}")
                    doc_logs.append(f"[Navigator] {curr_uid} -> RELEVANT")

                    # Graph expansion: add neighbors to stack
                    neighbors = []
                    neighbors.extend(book_env.get_semantic_neighbors(curr_idx, k=self.config.graph_k_neighbors))

                    # Add adjacent pages
                    if curr_idx + 1 < len(book_env.current_embeddings):
                        neighbors.append(curr_idx + 1)
                    if curr_idx - 1 >= 0:
                        neighbors.append(curr_idx - 1)

                    for n in neighbors:
                        if n in step_blacklist or n in step_accepted:
                            continue
                        if n in search_stack:
                            search_stack.remove(n)  # Move to top
                        search_stack.append(n)

                    doc_logs.append(f"  > Expanded neighbors. Stack Top: {[book_env.index_to_uid(i) for i in search_stack[-3:]]}")

                doc_logs.append(f"  [State] Accepted: {[book_env.index_to_uid(i) for i in step_accepted]}, Blacklist: {len(step_blacklist)}")

            doc_logs.append(f"[Step End] Evidence Pool Size: {len(global_evidence_pool)}")
            doc_logs.append(f"[Step End] All Retrieved Pages: {[book_env.index_to_uid(i) for i in all_retrieved_pages]}")

        # ==================== Stage 3: Reasoner ====================

        # Sampling: ensure we have exactly top_k pages for reasoner
        agent_indices = list(all_retrieved_pages)
        filled_indices = []
        target_count = self.config.top_k

        if len(agent_indices) > target_count:
            # Too many: truncate
            agent_indices = sorted(agent_indices)[:target_count]
            doc_logs.append(f"[Sampling] Truncated to {target_count} pages")
        elif len(agent_indices) < target_count and global_score_accumulator is not None:
            # Too few: fill with high-scoring pages
            needed = target_count - len(agent_indices)
            sorted_score_idx = global_score_accumulator.argsort(descending=True).tolist()
            for idx in sorted_score_idx:
                if len(filled_indices) >= needed:
                    break
                if idx not in all_retrieved_pages:
                    filled_indices.append(idx)
            doc_logs.append(f"[Sampling] Filled {len(filled_indices)} pages from global scores")

        final_indices = sorted(agent_indices) + sorted(filled_indices)
        doc_logs.append(f"[Sampling] Final Pages: {[book_env.index_to_uid(i) for i in final_indices]}")

        # Construct reasoner input
        reasoner_content = [
            REASONER_SYSTEM_PROMPT,
            f"Question: {question}\nOptions: {options}",
            "--- Collected Text Evidence ---",
            "\n\n".join(global_evidence_pool),
            "--- Visual Evidence (Reference Pages) ---"
        ]

        for idx in final_indices:
            img_path = book_env.get_image_path(idx)
            if img_path:
                reasoner_content.append(f"Page {book_env.index_to_uid(idx)}")
                reasoner_content.append(Image.from_file(img_path))

        try:
            resp = await self.reasoner_client.create([UserMessage(content=reasoner_content, source="user")])
            pred_text = resp.content
        except Exception as e:
            pred_text = "Error"
            doc_logs.append(f"[Reasoner Error]: {e}")

        # Extract answer from reasoner output
        match = re.search(r"Final Answer:\s*([A-F])", pred_text, re.IGNORECASE)
        pred_ans = match.group(1).upper() if match else "N/A"

        if pred_ans == "N/A":
            # Fallback: find last option letter in text
            candidates = re.findall(r"\b([A-F])\b", pred_text.upper())
            if candidates:
                pred_ans = candidates[-1]
                doc_logs.append("[Reasoner] Warning: Used fallback answer extraction")

        doc_logs.append(f"[Reasoner] Prediction: {pred_ans}")

        return {
            "pred_answer": pred_ans,
            "retrieved_indices": final_indices,
            "logs": "\n".join(doc_logs)
        }
