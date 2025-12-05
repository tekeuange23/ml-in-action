from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

# Function to initialize a model
def initialize_model(model_id):
    return ChatOllama(
      model=model_id,
      validate_model_on_init=True,
      temperature=0.2,
    )


# Initialize models
gemma3_llm = initialize_model("gemma3:270m")
deepseekr1_llm = initialize_model("deepseek-r1:1.5b")
llama3_llm = initialize_model("llama3.2")

# Prompt templates
gemma3_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}\n{format_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

deepseekr1_template = PromptTemplate(
    template="System: {system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

llama3_template = PromptTemplate(
    template="<s>[INST]{system_prompt}\n{format_prompt}\n{user_prompt}[/INST]",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)


# General function to get AI response
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    
    return chain.invoke({
      'system_prompt':system_prompt, 
      'user_prompt':user_prompt, 
      'format_prompt':json_parser.get_format_instructions()
    })


# Model-specific response functions
def gemma3_response(system_prompt, user_prompt):
    return get_ai_response(gemma3_llm, gemma3_template, system_prompt, user_prompt)

def deepseekr1_response(system_prompt, user_prompt):
    return get_ai_response(deepseekr1_llm, deepseekr1_template, system_prompt, user_prompt)

def llama3_response(system_prompt, user_prompt):
    return get_ai_response(llama3_llm, llama3_template, system_prompt, user_prompt)