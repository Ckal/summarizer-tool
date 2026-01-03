import os
import gradio as gr
import warnings
from smolagents import CodeAgent, InferenceClientModel
import json

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Import our Text Summarizer Tool
from summarizer_tool import TextSummarizerTool

# Initialize the Text Summarizer Tool
summarizer_tool = TextSummarizerTool()

# Load HuggingFace token from environment variable if available
hf_token = os.environ.get("HF_TOKEN")

# Sample texts for quick testing
sample_texts = {
    "News Article": """
    The European Union has approved a landmark artificial intelligence law, establishing comprehensive 
    regulations for AI systems according to their potential risks. The regulation categorizes AI into 
    four risk levels: unacceptable risk, high risk, limited risk, and minimal risk. Systems deemed to 
    pose unacceptable risks, such as those using subliminal manipulation or social scoring, will be 
    banned. High-risk systems, including those used in critical infrastructure, education, employment, 
    and law enforcement, will face strict requirements before market entry. These requirements include 
    risk assessments, high-quality datasets, detailed documentation, human oversight, and transparency. 
    The law aims to ensure AI systems are safe, transparent, traceable, non-discriminatory, and 
    environmentally friendly, while fostering innovation and establishing Europe as a leader in 
    responsible AI development.
    """,
    
    "Scientific Paper Abstract": """
    Recent advancements in large language models (LLMs) have demonstrated remarkable capabilities 
    across various tasks. However, these models still face challenges with reasoning, factuality, 
    and potential biases. This paper introduces a novel framework for enhancing LLM performance 
    through a multi-stage processing pipeline that integrates retrieval-augmented generation, 
    self-reflection mechanisms, and external knowledge verification. Our approach, which we call 
    RACER (Retrieval-Augmented Chain-of-thought Enhanced Reasoning), demonstrates significant 
    improvements across benchmarks testing reasoning (GSM8K, +12.3%), factuality (FACTOR, +17.8%), 
    and reduced bias (BBQ, -24.5%) compared to base models. Additionally, we show that RACER is 
    complementary to existing techniques like chain-of-thought prompting and can be applied to 
    various model architectures with minimal computational overhead. Through extensive ablation 
    studies, we identify the contribution of each component and provide insights for efficient 
    implementation in real-world applications.
    """,
    
    "Business Report": """
    Q1 Financial Performance Summary: The company achieved significant growth in the first quarter, 
    with revenue reaching $78.5 million, a 24% increase compared to the same period last year. This 
    growth was primarily driven by our expanded product portfolio and increased market penetration in 
    European and Asian markets. Our flagship product line saw sales increase by 32%, while our new 
    service offerings contributed $12.3 million in revenue. Gross margin improved to 62.8% from 58.4% 
    in the previous year, reflecting our successful cost optimization initiatives and economies of scale. 
    Operating expenses were $28.7 million, up 15% year-over-year, primarily due to increased R&D 
    investments and marketing campaigns for new product launches. Despite these investments, operating 
    profit grew by 42% to $20.5 million, representing a 26.1% operating margin. Our customer base expanded 
    by 15%, with particularly strong growth in the enterprise segment. Looking ahead, we maintain our 
    full-year guidance of $320-340 million in revenue and anticipate continued margin improvement as we 
    scale operations.
    """
}

# Function to directly use the summarizer tool
def summarize_text(text, model, max_length, min_length, style):
    try:
        # Convert max_length and min_length to integers
        max_length = int(max_length) if max_length else None
        min_length = int(min_length) if min_length else None
        
        # Call the summarizer tool
        result = summarizer_tool(
            text=text,
            model=model,
            max_length=max_length,
            min_length=min_length,
            style=style
        )
        return result
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

# Function to use the summarizer with an agent (if token available)
def agent_summarize(text, instruction, temperature=0.7):
    if not hf_token:
        return "Agent summarization requires a HuggingFace API token. Please set the HF_TOKEN environment variable."
        
    if not text or not instruction:
        return "Please provide both text and instructions."
        
    try:
        # Initialize model for agent
        model = InferenceClientModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            token=hf_token,
            temperature=float(temperature)
        )
        
        # Create the agent with our summarizer tool
        agent = CodeAgent(tools=[summarizer_tool], model=model)
        
        # Format the prompt with the instruction and text
        prompt = f"Instruction: {instruction}\n\nText to summarize: {text}"
        
        # Run the agent
        result = agent.run(prompt)
        return result
    except Exception as e:
        return f"Error with agent summarization: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Advanced Text Summarizer") as demo:
    gr.Markdown("# 📝 Advanced Text Summarizer")
    gr.Markdown("Summarize text using different models and styles, with optional agent assistance.")
    
    with gr.Tabs():
        # Direct summarization tab
        with gr.Tab("Direct Summarization"):
            with gr.Row():
                with gr.Column():
                    # Input section
                    direct_text_input = gr.Textbox(
                        label="Text to Summarize", 
                        placeholder="Enter text to summarize...",
                        lines=10
                    )
                    
                    # Sample texts dropdown
                    sample_dropdown = gr.Dropdown(
                        choices=list(sample_texts.keys()),
                        label="Or Select a Sample Text"
                    )
                    
                    # Configuration options
                    with gr.Row():
                        with gr.Column():
                            model_dropdown = gr.Dropdown(
                                choices=list(summarizer_tool.available_models.keys()),
                                value="facebook/bart-large-cnn",
                                label="Summarization Model"
                            )
                            
                            style_dropdown = gr.Dropdown(
                                choices=["concise", "detailed", "bullet_points"],
                                value="concise",
                                label="Summary Style"
                            )
                            
                        with gr.Column():
                            max_length_slider = gr.Slider(
                                minimum=50, 
                                maximum=250, 
                                value=130, 
                                step=10,
                                label="Maximum Summary Length"
                            )
                            
                            min_length_slider = gr.Slider(
                                minimum=10, 
                                maximum=100, 
                                value=30, 
                                step=5,
                                label="Minimum Summary Length"
                            )
                    
                    direct_summarize_button = gr.Button("Summarize Text")
                
                with gr.Column():
                    # Output section
                    direct_output = gr.Textbox(label="Summary", lines=12)
        
        # Agent-assisted summarization tab
        with gr.Tab("Agent-Assisted Summarization"):
            with gr.Row():
                with gr.Column():
                    # Input section
                    agent_text_input = gr.Textbox(
                        label="Text to Summarize", 
                        placeholder="Enter text to summarize...",
                        lines=10
                    )
                    
                    # Agent instruction
                    instruction_input = gr.Textbox(
                        label="Instructions for the Agent",
                        placeholder="E.g., 'Summarize this text and highlight the three most important points'",
                        lines=2,
                        value="Summarize this text in a professional tone, highlighting key information."
                    )
                    
                    # Sample instructions
                    instruction_examples = gr.Dropdown(
                        choices=[
                            "Summarize this text in a professional tone, highlighting key information.",
                            "Create a very concise summary focusing only on actionable items.",
                            "Summarize this for a high school student, explaining complex terms.",
                            "Extract the main argument and supporting evidence from this text.",
                            "Create a summary that focuses on financial implications mentioned in the text."
                        ],
                        label="Or Select Example Instructions"
                    )
                    
                    # Temperature setting
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Agent Temperature (creativity)"
                    )
                    
                    agent_summarize_button = gr.Button("Use Agent to Summarize")
                
                with gr.Column():
                    # Output section
                    agent_output = gr.Textbox(label="Agent Response", lines=15)
                    
    # Set up event handlers
    def load_sample(sample_name):
        return sample_texts.get(sample_name, "")
    
    def load_instruction(instruction):
        return instruction
    
    sample_dropdown.change(
        load_sample,
        inputs=sample_dropdown,
        outputs=direct_text_input
    )
    
    instruction_examples.change(
        load_instruction,
        inputs=instruction_examples,
        outputs=instruction_input
    )
    
    direct_summarize_button.click(
        summarize_text,
        inputs=[direct_text_input, model_dropdown, max_length_slider, min_length_slider, style_dropdown],
        outputs=direct_output
    )
    
    agent_summarize_button.click(
        agent_summarize,
        inputs=[agent_text_input, instruction_input, temperature_slider],
        outputs=agent_output
    )
    
    # Also allow using sample text for agent tab
    sample_dropdown.change(
        load_sample,
        inputs=sample_dropdown,
        outputs=agent_text_input
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()