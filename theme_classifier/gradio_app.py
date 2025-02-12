import gradio as gr
import matplotlib.pyplot as plt
from theme_classifier import ThemeClassifier
import os
import pandas as pd
from character_network import name_entity_recognizer, character_network_generator

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

def get_character_network(subtitles_path, ner_path):
    ner = name_entity_recognizer.NamedEntityRecognizer()
    if not ner_path:
        ner_path = "temp.csv"
    ner_data = ner.get_ners(subtitles_path, ner_path)

    character_network = character_network_generator.CharacterNetworkGenerator()
    relationship_data = character_network.generate_character_network(ner_data)
    html = character_network.draw_network_graph(relationship_data)

    return html

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

        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtutles or Script Path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path], outputs=[network_html])


    iface.launch(share=True)

if __name__ == '__main__':
    main()
