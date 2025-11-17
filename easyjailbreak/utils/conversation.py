"""
Simple conversation template class to replace fast-chat dependency.
This provides a minimal interface compatible with fast-chat's Conversation class.
"""


class SimpleConversation:
    """
    A simple conversation template class that mimics fast-chat's Conversation interface.
    
    This class provides:
    - roles: tuple of role names (e.g., ("user", "assistant"))
    - messages: list of (role, content) tuples
    - append_message(role, content): add a message to the conversation
    - to_openai_api_messages(): convert to OpenAI API format
    """
    
    def __init__(self, name="chatgpt", system_message="You are a helpful assistant.", roles=("user", "assistant")):
        """
        Initialize a simple conversation template.
        
        :param str name: Template name (for compatibility)
        :param str system_message: System message (currently not used in MJP)
        :param tuple roles: Role names tuple, e.g., ("user", "assistant")
        """
        self.name = name
        self.system_message = system_message
        self.roles = roles
        self.messages = []
        self.offset = 0  # For compatibility with fast-chat
    
    def append_message(self, role, content):
        """
        Append a message to the conversation.
        
        :param str role: Role name (e.g., "user" or "assistant")
        :param str content: Message content
        """
        self.messages.append((role, content))
    
    def to_openai_api_messages(self):
        """
        Convert the conversation to OpenAI chat completion format.
        
        :return list: List of dicts with "role" and "content" keys
        """
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]
        
        for i, (role, content) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": content})
            else:
                if content is not None:
                    ret.append({"role": "assistant", "content": content})
        return ret


def get_conv_template(template_name="chatgpt"):
    """
    Get a conversation template by name.
    
    This function replaces fast-chat's get_conv_template.
    Currently only supports "chatgpt" template.
    
    :param str template_name: Template name (default: "chatgpt")
    :return SimpleConversation: A conversation template instance
    """
    if template_name == "chatgpt":
        return SimpleConversation(
            name="chatgpt",
            system_message="You are a helpful assistant.",
            roles=("user", "assistant")
        )
    else:
        # Fallback: return a generic template
        return SimpleConversation(
            name=template_name,
            system_message="You are a helpful assistant.",
            roles=("user", "assistant")
        )

