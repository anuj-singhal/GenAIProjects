
class Prompts:
    system_prompt = """You are an assistant that analyzes the contents\n
    and perform the specific tasks.
    """
    user_prompt = """Perform the following tasks."""

    def __init__(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def user_prompt_add(self, message):
        self.user_prompt += message
        return self.user_prompt
    
    def get_messages(self):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt}
        ]
        
        
