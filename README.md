# Retention Order Support Vector Machine (ROSVM)

ROSVM is a [Ranking Support Vector Machine (RankSVM)](https://en.wikipedia.org/wiki/Ranking_SVM) implementation for retention order prediction of liquid chromatography (LC) retention times (RT). It was initally proposed by [Bach et al. (2018)](https://academic.oup.com/bioinformatics/article/34/17/i875/5093227). 

This library aims to be a more self-contained implementation, allowing the user to easily train models and make predictions. 

# Install

## Using conda in a new environment

1) Create a new [conda](https://docs.conda.io/en/latest/miniconda.html) environment using:
    ```bash
    conda env create --file conda/environment.yml
    ```

2) Active the environment:
    ```bash
    conda activate rosvm
    ```
   
3) Install the package into the environment:
    ```bash
    pip install . 
    ```

4) (optional) Use the environment in Jupyter notebooks to run [the examples](rosvm/ranksvm/tutorial):
    1) Install the IPython kernel:
    
        ```bash
        conda install ipykernel
        ```

    2) Make the environment available as notebook kernel:
    
        ```bash
        python -m ipykernel install --user --name=rosvm
        ```

## Using pip

You can install the package directly using:
```bash
pip install . 
```
However, the installation of [rdkit](https://github.com/rdkit/rdkit) can be a bit tricky. You can find installation instructions for various operating systems [here](https://github.com/rdkit/rdkit/blob/master/Docs/Book/Install.md).  



# Citation

If you are using the library please cite: 

- For the general approach of retention order prediction 

```bibtex
@article{Bach2018,
    author = {Bach, Eric and Szedmak, Sandor and Brouard, Céline and Böcker, Sebastian and Rousu, Juho},
    title = "{Liquid-chromatography retention order prediction for metabolite identification}",
    journal = {Bioinformatics},
    volume = {34},
    number = {17},
    pages = {i875-i883},
    year = {2018},
    month = {09},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/bty590},
    url = {https://doi.org/10.1093/bioinformatics/bty590},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/34/17/i875/25702364/bty590.pdf},
}
```

- For the actual ROSVM implementation 

```bibtex
@software{Bach_Retention_Order_Support_2020,
    author = {Bach, Eric},
    month = {5},
    title = {{Retention Order Support Vector Machine (ROSVM)}},
    url = {https://github.com/bachi55/rosvm},
    version = {0.4.0},
    year = {2020}
}
```
