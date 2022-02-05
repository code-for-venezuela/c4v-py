# Configuration
The main configuration module in `c4v-py` is `c4v.config`. To access all configuration variables, you can use code like this:

```python
from c4v.config import settings

print(settings.<config variable>)
```

Where `<config variable>` could be one of the following:

- `DATE_FORMAT` : a `datetime` compatible format specifying how dates are stored. **Defaults to `%Y-%m-%d %H:%M:%S.%f%z`**
- `C4V_FOLDER` : folder where to store the `.c4v` files folder, required to store persistent data such as the database, metadata and models. ***Defaults to `$HOME`***
- `LOCAL_SQLITE_DB_NAME` : name for the local sqlite database, **defaults to `c4v_db.sqlite`**
- `LOCAL_SQLITE_DB` : Path to the local sqlite database file. **Defaults to `c4v_folder`**
- `DEFAULT_BASE_LANGUAGE_MODEL` : default base language model used to train more models in the downstream task of classification.
- `DEFAULT_LANG_MODEL_MIN_LOSS` : Minimum acceptable loss for base language model, if the loss is greater, a new training is required. **Defaults to 0.15**
- `EXPERIMENTS_DIR` : Where to store experiments. **Defaults to `<c4v_folder>/experiments`**
- `PERSISTENCY_MANAGER` : type of persistency manager to use. One of the following:
    - `SQLITE` : the **default** SQLite based persistency manager
    - `GCLOUD`  : use a google cloud based persistency manager, measurements are stored in cloud
    - `USER` : Use the user defined persistency manager, specified by the USER_PERSISTENCY_MANAGER_PATH setting.
- `USER_PERSISTENCY_MANAGER_PATH` : Path for a custom user persistency manager python file. Should export a function: `get_persistency_manager : () -> BasePersistencyManager`
- `USER_PERSISTENCY_MANAGER_MODULE` : Module starting from the file specified in  `USER_PERSISTENCY_MANAGER_PATH` where to find the function
- `CLI_LOGGING_LEVEL` : specify which messages will be logged by the CLI. 
    - `10`: error, warn, success, info
    - `9`:  error, warn, success
    - `8`:  error, warn
    - `7`:  error
    - `[1-6]`:  reserved for future use
    - `0`: No logging 
- `STORAGE_BUCKET` : Used to choose the name of the bucket where data like the classifier models is stored in google cloud. Might require the `gcloud` installation profile.
- `SCRAPED_DATA_TABLE` : Used to select the table where to store and retrieve data in  google cloud. Might require the `gcloud` installation profile.
- `SCRAPING_CLOUD_URL_TRIGGER` : Url to use to trigger a scaping process in a google cloud function. Might require the `gcloud` installation profile.
- `CRAWLING_CLOUD_URL_TRIGGER` : Url to use to trigger a crawling process in a google cloud function. Might require the `gcloud` installation profile.
- `CLASSIFY_CLOUD_URL_TRIGGER` : Url to use to trigger a classification process in a google cloud function.  Might require the `gcloud` installation profile.
- `GCLOUD_PROJECT_ID` : project id for the gcloud project, where all the cloud operations will be performed. Might require the `gcloud` installation profile.
- `GCLOUD_MAX_CONTENT_LEN` : Maximum length for the content field in google cloud, might be truncated if it's longer than the maximum length in the big query table.

Also, you can set the value for any of the previously mentioned configuration variables by using them as environment variables and adding the prefix `"C4V_"`. For example:

```bash
export C4V_DATE_FORMAT="my new date format"
```
