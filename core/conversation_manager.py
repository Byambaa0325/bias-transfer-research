"""
Conversation Manager

Handles multi-turn conversation reconstruction and tracking for bias injection experiments.
Manages nested conversation history from multiple bias injection rounds.
"""

from typing import Dict, Any, List, Optional


class ConversationManager:
    """
    Manages multi-turn conversation reconstruction and tracking.

    Responsibilities:
    - Reconstruct conversation history from nested structure
    - Track bias injection count across conversation rounds
    - Convert between different conversation formats

    Example:
        >>> manager = ConversationManager()
        >>> history = {
        ...     'turn1_question': 'Question A',
        ...     'turn1_response': 'Response A',
        ...     'previous_conversation': {
        ...         'turn1_question': 'Question B',
        ...         'turn1_response': 'Response B'
        ...     }
        ... }
        >>> messages = manager.reconstruct_from_history(history)
        >>> len(messages)  # 4 messages (2 from previous, 2 from current)
        4
    """

    @staticmethod
    def reconstruct_from_history(
        existing_conversation: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Recursively reconstruct conversation from nested structure.

        Only includes priming turns (Turn 1) from previous nodes.
        The original prompt is excluded - it should only be asked at the end.

        This supports nested bias injection where multiple biases are layered:
        Round 1: Availability bias priming → response
        Round 2: Confirmation bias priming → response
        Round 3: Original question → measured response

        Args:
            existing_conversation: Previous conversation dictionary with nested structure
                Format: {
                    'turn1_question': str,
                    'turn1_response': str,
                    'previous_conversation': dict (optional, recursive)
                }

        Returns:
            List of message dictionaries for the API
                Format: [{"role": "user", "content": "..."}, ...]

        Example:
            >>> manager = ConversationManager()
            >>> nested_conv = {
            ...     'turn1_question': 'Recent question',
            ...     'turn1_response': 'Recent response',
            ...     'previous_conversation': {
            ...         'turn1_question': 'Old question',
            ...         'turn1_response': 'Old response'
            ...     }
            ... }
            >>> messages = manager.reconstruct_from_history(nested_conv)
            >>> # Result: [
            >>> #   {"role": "user", "content": "Old question"},
            >>> #   {"role": "assistant", "content": "Old response"},
            >>> #   {"role": "user", "content": "Recent question"},
            >>> #   {"role": "assistant", "content": "Recent response"}
            >>> # ]
        """
        messages = []

        def _reconstruct_recursive(
            conv_dict: Optional[Dict[str, Any]],
            messages_list: List[Dict[str, str]]
        ):
            """
            Inner recursive function to rebuild conversation.

            Processes conversations in chronological order (oldest first).
            """
            if not conv_dict:
                return

            # Process previous conversation first (recursive, depth-first)
            if conv_dict.get('previous_conversation'):
                _reconstruct_recursive(
                    conv_dict['previous_conversation'],
                    messages_list
                )

            # Add priming turns (Turn 1) from this level
            if conv_dict.get('turn1_question'):
                messages_list.append({
                    "role": "user",
                    "content": conv_dict['turn1_question']
                })

            if conv_dict.get('turn1_response'):
                messages_list.append({
                    "role": "assistant",
                    "content": conv_dict['turn1_response']
                })

            # Note: We intentionally skip 'original_prompt' and 'turn2_response'
            # These are only relevant for the final (current) bias injection round

        _reconstruct_recursive(existing_conversation, messages)
        return messages

    @staticmethod
    def get_bias_count(existing_conversation: Optional[Dict[str, Any]]) -> int:
        """
        Get the number of bias injections in conversation history.

        Each bias injection adds one round of priming (Turn 1 + response).
        The count represents the depth of nested bias injection.

        Args:
            existing_conversation: Conversation history dictionary

        Returns:
            Number of bias injection rounds (0 if no history)

        Example:
            >>> manager = ConversationManager()
            >>> conv = {'bias_count': 2}
            >>> manager.get_bias_count(conv)
            2
            >>> manager.get_bias_count(None)
            0
        """
        if not existing_conversation:
            return 0
        return existing_conversation.get('bias_count', 0)

    @staticmethod
    def create_conversation_dict(
        turn1_question: str,
        turn1_response: str,
        original_prompt: str,
        turn2_response: str,
        previous_conversation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a conversation dictionary with metadata.

        Args:
            turn1_question: Priming question (Turn 1)
            turn1_response: Model's response to priming
            original_prompt: The actual question/prompt
            turn2_response: Model's response to original prompt
            previous_conversation: Previous conversation (if nested)

        Returns:
            Conversation dictionary with all turns and metadata

        Example:
            >>> manager = ConversationManager()
            >>> conv = manager.create_conversation_dict(
            ...     turn1_question="Priming question?",
            ...     turn1_response="Priming response",
            ...     original_prompt="Original question?",
            ...     turn2_response="Final answer"
            ... )
            >>> conv['bias_count']
            1
        """
        conversation_dict = {
            'turn1_question': turn1_question,
            'turn1_response': turn1_response,
            'original_prompt': original_prompt,
            'turn2_response': turn2_response
        }

        # Track nesting depth
        if previous_conversation:
            conversation_dict['previous_conversation'] = previous_conversation
            bias_count = ConversationManager.get_bias_count(previous_conversation) + 1
            conversation_dict['bias_count'] = bias_count
        else:
            conversation_dict['bias_count'] = 1

        return conversation_dict

    @staticmethod
    def convert_to_messages(
        conversation_dict: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Convert conversation dictionary to messages list format.

        Includes all turns including the original prompt and final response.
        Use this to get the full conversation for display/analysis.

        Args:
            conversation_dict: Full conversation dictionary

        Returns:
            Complete list of messages

        Example:
            >>> manager = ConversationManager()
            >>> conv = {
            ...     'turn1_question': 'Q1',
            ...     'turn1_response': 'R1',
            ...     'original_prompt': 'Q2',
            ...     'turn2_response': 'R2'
            ... }
            >>> messages = manager.convert_to_messages(conv)
            >>> len(messages)
            4
        """
        # Get priming turns (all previous rounds)
        messages = ConversationManager.reconstruct_from_history(conversation_dict)

        # Add current round's full conversation
        if conversation_dict.get('turn1_question'):
            messages.append({
                "role": "user",
                "content": conversation_dict['turn1_question']
            })

        if conversation_dict.get('turn1_response'):
            messages.append({
                "role": "assistant",
                "content": conversation_dict['turn1_response']
            })

        if conversation_dict.get('original_prompt'):
            messages.append({
                "role": "user",
                "content": conversation_dict['original_prompt']
            })

        if conversation_dict.get('turn2_response'):
            messages.append({
                "role": "assistant",
                "content": conversation_dict['turn2_response']
            })

        return messages

    @staticmethod
    def print_conversation_tree(
        conversation_dict: Dict[str, Any],
        indent: int = 0
    ):
        """
        Print conversation history as a tree (for debugging).

        Args:
            conversation_dict: Conversation dictionary
            indent: Current indentation level

        Example:
            >>> manager = ConversationManager()
            >>> manager.print_conversation_tree(nested_conv)
            Bias Injection #1:
              Q: Old question
              A: Old response
              Bias Injection #2:
                Q: Recent question
                A: Recent response
        """
        prefix = "  " * indent

        # Print previous conversation first (recursive)
        if conversation_dict.get('previous_conversation'):
            ConversationManager.print_conversation_tree(
                conversation_dict['previous_conversation'],
                indent
            )

        # Print current level
        bias_num = conversation_dict.get('bias_count', 1)
        print(f"{prefix}Bias Injection #{bias_num}:")

        if conversation_dict.get('turn1_question'):
            print(f"{prefix}  Q: {conversation_dict['turn1_question'][:50]}...")

        if conversation_dict.get('turn1_response'):
            print(f"{prefix}  A: {conversation_dict['turn1_response'][:50]}...")
