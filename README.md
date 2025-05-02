# Predicting Gene Expression levels using Custom Transformer

A modular and transparent implementation of Transformer-based architectures built from the ground up using Python. This project includes custom attention mechanisms, encoder-decoder structures, and integrates XAI (Explainable AI) techniques for better interpretability.

## Features

- 🧠 **Custom Transformer Architecture**: All components written from scratch, no external deep learning frameworks.
- 🧩 **Modular Design**: Easily extensible attention layers, feedforward networks, and projection modules.
- 📊 **Explainable AI (XAI)**: Integration of XAI tools to visualize and interpret model decisions.
- 📚 **Notebooks**: Training and visualization notebooks for understanding and debugging.
- 🧪 **Clean Training Scripts**: Includes `Trainner.ipynb` for training and experimentation.

## Project Structure

```
.
├── Architectures/      # Core architecture components (perceiver)
├── Data/               #  Data directory
├── layers/             # Custom layers (attention, feedforward, etc.)
├── images/             # Plots or architecture diagrams
├── XAI/                # Explainable AI related tools
├── Trainner.ipynb      # Training notebook
├── example.ipynb       # Sample usage notebook
├── plotdata.ipynb      # Data visualization
├── xai.ipynb           # XAI experiments
```


## Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/TransformerFromScratch.git
   cd TransformerFromScratch
2. Download the dataset for human
   ```bash
   !wget -r -np -nH --reject "index.html*" --cut-dirs 6 \
   https://krishna.gs.washington.edu/content/members/vagar/Xpresso/data/datasets/pM10Kb_1KTest/
3. Open and run the notebooks to explore the model and its explainability (ongoing work).

## Ongoing Work
At Current, i am working on predicting mRNA levels with Protein levels, further i would be using XAI techniqes to investigate the prediction of these expression levels

## Author and Refrences
This reprository is created by **Mohit\Se00n00** however the work is havily influenced by following research papers.

## Related Papers

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Perceiver: General Perception with Iterative Attention (Jaegle et al., 2021)](https://arxiv.org/abs/2103.03206)
- [Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks (Vikram et al., 2020)](https://doi.org/10.1016/j.celrep.2020.107663)
- [Predicting gene and protein expression levels from DNA and protein sequences with Perceiver (Matteo et al., 2023)](https://doi.org/10.1016/j.cmpb.2023.107504)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contributions are welcome!
