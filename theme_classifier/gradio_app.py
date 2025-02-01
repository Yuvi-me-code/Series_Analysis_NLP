import gradio as gr
import matplotlib.pyplot as plt
from theme_classifier import ThemeClassifier
import os
import pandas as pd

def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list_str = ",".join(word.strip().title() for word in theme_list_str.replace(" ", "").split(","))
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_data = theme_classifier.get_themes(subtitles_path,save_path)

    # Remove dialogue from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_data = output_data[theme_list]

    output_data = output_data[theme_list].sum().reset_index()
    output_data.columns = ['Theme','Score']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(output_data["Theme"], output_data["Score"], color="skyblue")
    ax.set_xlabel("Score")
    ax.set_ylabel("Theme")
    ax.set_title("Series' Themes")

    return fig  

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Anime Theme Classification</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.Plot(label="Theme Analysis")  # Use gr.Plot instead of gr.BarPlot
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes (comma-separated)")
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path (optional)")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])

    iface.launch(share=False)

if __name__ == '__main__':
    main()
