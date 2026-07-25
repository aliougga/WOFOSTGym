Searched for files: *input*
Listed directory fileinput
Searched for "YAMLCropDataProvider"
Viewed wofost_base.py:1-800
Viewed __init__.py:1-10

Les fichiers utilisés pour gérer les inputs du simulateur PCSE (Python Crop Simulation Environment) dans votre projet se situent principalement à deux niveaux :

### 1. Au niveau de la librairie PCSE (Les loaders)
Ces fichiers s'occupent de parser et charger les données d'entrée depuis les fichiers YAML. Ils sont situés dans le dossier `c:\Projects\WOFOSTGym\pcse\pcse\fileinput\` :
*   `yaml_cropdataprovider.py` : Utilisé pour charger les paramètres liés à la culture (`YAMLCropDataProvider`).
*   `yaml_soildataprovider.py` : Utilisé pour charger les paramètres liés au sol (`YAMLSoilDataProvider`).
*   `yaml_agro_loader.py` : Utilisé pour charger les données d'agromanagement (`YAMLAgroManagementReader`).

*(Il y a également la gestion de la météo qui ne provient pas des fichiers YAML mais de l'API de base via `NASAPowerWeatherDataProvider` situé directement dans `pcse`).*

### 2. Au niveau de l'environnement Gym (`pcse_gym`)
C'est le fichier de base de votre environnement qui se charge de faire l'appel à ces loaders pour fournir les inputs lors de l'initialisation du moteur WOFOST. Ce fichier se trouve ici :
*   `c:\Projects\WOFOSTGym\pcse_gym\pcse_gym\envs\wofost_base.py`

Si vous regardez dans la méthode `__init__` (vers la ligne **95**), vous verrez comment ces inputs sont instanciés :
```python
# Load all model parameters from .yaml files
self.crop = pcse.fileinput.YAMLCropDataProvider(fpath=os.path.join(base_fpath, crop_fpath))
self.soil = pcse.fileinput.YAMLSoilDataProvider(fpath=os.path.join(base_fpath, soil_fpath))
self.parameterprovider = pcse.base.ParameterProvider(soildata=self.soil, cropdata=self.crop)

# Agromanagement data
self.agromanagement = self._load_agromanagement_data(os.path.join(base_fpath, agro_fpath))

# Weather data provider
self.weatherdataprovider = NASAPowerWeatherDataProvider(*self.location)
```
Ces providers (`self.parameterprovider`, `self.weatherdataprovider` et `self.agromanagement`) sont ensuite passés directement au moteur : `Wofost8Engine(...)`.