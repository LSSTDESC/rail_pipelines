import copy
import os
import sys
import glob
import subprocess
from pathlib import Path
import pprint
import time
import functools
import itertools

import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import yaml

from .pipe_options import RunMode
from . import name_utils

class RailProject:
    config_template = {
        "IterationVars": {},
        "CommonPaths": {},
        "PathTemplates": {},
        "Catalogs": {},
        "Files": {},
        "Pipelines": {},
        "Flavors": {},
        "Selections": {},
        "PZAlgorithms": {},
        "NZAlgorithms": {},
    }
 

    def __init__(self, name, config_dict):
        self.name = name
        self._config_dict = config_dict
        self.config = copy.deepcopy(self.config_template)
        for k in self.config.keys():
            if (v := self._config_dict.get(k)) is not None:
                self.config[k] = v
        # self.interpolants = self.get_interpolants()
        self.name_factory = name_utils.NameFactory(
            config=self.config,
            templates=config_dict.get('PathTemplates', {}),
            interpolants=self.config.get("CommonPaths", {})
        )
        self.name_factory.resolve_from_config(
            self.config.get("CommonPaths")
        )

    def __repr__(self):
        return f"{self.name}"

    @staticmethod
    def load_config(config_file):
        project_name = Path(config_file).stem
        with open(config_file, "r") as fp:
            config_dict = yaml.safe_load(fp)
        project = RailProject(project_name, config_dict)
        # project.resolve_common()
        return project

    def get_path_templates(self):
        """ Return the dictionary of templates used to construct paths """
        return self.name_factory.get_path_templates()
    
    def get_path(self, path_key, **kwargs):
        """ Resolve and return a path using the kwargs as interopolants """
        return self.name_factory.resolve_path_template(path_key, **kwargs)    

    def get_common_paths(self):
        """ Return the dictionary of common paths """        
        return self.name_factory.get_common_paths()
    
    def get_common_path(self, path_key, **kwargs):
        """ Resolve and return a common path using the kwargs as interopolants """       
        return self.name_factory.resolve_common_path(path_key, **kwargs)    

    def get_files(self):
        """ Return the dictionary of specific files """        
        return self.config.get("Files")

    def get_file(self, name, **kwargs):
        """ Resolve and return a file using the kwargs as interpolants """
        files = self.get_files()
        file_dict = files.get(name, None)
        if file_dict is None:
            raise ValueError(f"file '{name}' not found in {self}")
        path = self.name_factory.resolve_path(file_dict, "PathTemplate", **kwargs)
        return path

    def get_flavors(self):
        """ Return the dictionary of analysis flavor variants """
        flavors = self.config.get("Flavors")
        baseline = flavors.get("baseline", {})
        for k, v in flavors.items():
            if k != "baseline":
                flavors[k] = baseline | v

        return flavors

    def get_flavor(self, name):
        """ Resolve the configuration for a particular analysis flavor variant """
        flavors = self.get_flavors()
        flavor = flavors.get(name, None)
        if flavor is None:
            raise ValueError(f"flavor '{name}' not found in {self}")
        return flavor

    def get_file_for_flavor(self, flavor, label, **kwargs):
        """ Resolve the file associated to a particular flavor and label 

        E.g., flavor=baseline and label=train would give the baseline training file
        """       
        flavor_dict = self.get_flavor(flavor)
        try:
            file_alias = flavor_dict['FileAliases'][label]
        except KeyError as msg:
            raise ValueError(f"Label '{label}' not found in flavor '{flavor}'")                           
        return self.get_file(file_alias, flavor=flavor, label=label, **kwargs)

    def get_file_metadata_for_flavor(self, flavor, label):
        """ Resolve the metadata associated to a particular flavor and label 
        
        E.g., flavor=baseline and label=train would give the baseline training metadata
        """
        flavor_dict = self.get_flavor(flavor)
        try:
            file_alias = flavor_dict['FileAliases'][label]
        except KeyError as msg:
            raise ValueError(f"Label '{label}' not found in flavor '{flavor}'")                           
        return self.get_files()[file_alias]
    
    def get_selections(self):
        return self.config.get("Selections")

    def get_selection(self, name):
        selections = self.get_selections()
        selection = selections.get(name, None)
        if selection is None:
            raise ValueError(f"selection '{name}' not found in {self}")
        return selection

    def get_pzalgorithms(self):
        return self.config.get("PZAlgorithms")

    def get_pzalgorithm(self, name):
        pzalgorithms = self.get_pzalgorithms()
        pzalgorithm = pzalgorithms.get(name, None)
        if pzalgorithm is None:
            raise ValueError(f"pz algorithm '{name}' not found in {self}")
        return pzalgorithm

    def get_nzalgorithms(self):
        return self.config.get("NZAlgorithms")

    def get_nzalgorithm(self, name):
        nzalgorithms = self.get_nzalgorithms()
        nzalgorithm = nzalgorithms.get(name, None)
        if nzalgorithm is None:
            raise ValueError(f"nz algorithm '{name}' not found in {self}")
        return nzalgorithm

    def get_catalogs(self):
        return self.config['Catalogs']
    
    def get_catalog(self, catalog, **kwargs):
        catalog_dict = self.config['Catalogs'].get(catalog, {})
        path = self.name_factory.resolve_path(catalog_dict, "PathTemplate", **kwargs)

        return path

    def get_pipelines(self):
        return self.config.get("Pipelines")

    def get_pipeline(self, name, **kwargs):
        pipelines = self.get_pipelines()
        return pipelines.get(name, None)

    def get_flavor_args(self, flavors):
        flavor_dict = self.get_flavors()
        if 'all' in flavors:
            return list(flavor_dict.keys())
        return flavors
    
    def get_selection_args(self, selections):        
        selection_dict = self.get_selections()
        if 'all' in selections:
            return list(selection_dict.keys())
        return selections

    def generate_kwargs_iterable(self, **iteration_dict):
        iteration_vars = list(iteration_dict.keys())
        iterations = itertools.product( 
            *[
                iteration_dict.get(key) for key in iteration_vars
            ]
        )
        iteration_kwarg_list = []
        for iteration_args in iterations:
            iteration_kwargs = {
                iteration_vars[i]: iteration_args[i]
                for i in range(len(iteration_vars))
            }
            iteration_kwarg_list.append(iteration_kwargs)
        return iteration_kwarg_list

    def generate_ceci_command(
        self,
        pipeline_path,
        config=None,    
        inputs=None,
        output_dir='.',
        log_dir='.',
        **kwargs,
    ):

        if config is None:
            config = pipeline_path.replace('.yaml', '_config.yml')
            
        command_line = [
            f"ceci",
            f"{pipeline_path}",
            f"config={config}",
            f"output_dir={output_dir}",
            f"log_dir={log_dir}",
        ]

        for key, val in inputs.items():
            command_line.append(f"inputs.{key}={val}")

        
        for key, val in kwargs.items():
            command_line.append(f"{key}={val}")

        return command_line
