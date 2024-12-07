# Code for the Ordered Matrix Dirichlet

Niklas Stoehr (ETH Zurich), Benjamin Radford (UNC Charlotte), Ryan Cotterell (ETH Zurich), Aaron Schein (University of Chicago)<br><br>


AISTATS: https://proceedings.mlr.press/v206/stoehr23a/stoehr23a.pdf <br>
arXiv: https://arxiv.org/pdf/2212.04130.pdf

## Folder structure and installation

When running the code locally, you have to install the user-defined modules ``omd0configs``,``omd1data``,``omd2model``. From the root, install the modules by executing

```
pip install -e omd0configs
pip install -e omd1data
pip install -e omd2model
```

``omd0configs`` features configuration methods and other helper functions <br>
``omd1data`` features different data loading functionality <br>
``omd2model`` features different models <br>

In addition, we recommend installing the requirements listed in the requirements.txt file:

```
pip install -r requirements.txt
```

## Data

This repository offers functionalites for two kinds of data: *synthetic data* and *real-world conflict data*. For the latter, you need to download 
the freely accessible [*ICEWS coded event data*](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075) from the Harvard Dataverse. 
In particular, we recommend downloading the event data from 2015 to 2020:

```
events.2020.20220623.tab.zip
events.2019.20200427085336.tab
events.2018.20200427084805.tab
events.2017.20201119.zip
events.2016.20180710092843.tab
events.2015.20180710092545.tab
```
Place the data at ``data/conflict/icews``.