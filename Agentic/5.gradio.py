import gradio as gr


def process_text(text):
  return f"You entered: '{text}'"

demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=gr.Textbox(label="Output"),
    title="Simple Text Echoer",
    description="A simple Gradio app that echoes the text you enter."
)

if __name__ == "__main__":
    demo.launch()

# import gradio as gr

# def greet(name, intensity):
#   return "Hello, " + name + "!" * int(intensity)

# demo = gr.Interface(
#   fn=greet,
#   inputs=["text", "slider"],
#   outputs=["text"],
# )

# demo.launch(server_name="127.0.0.1", server_port=5000)