# Your custom persistency manager
Maybe you want to store the data gathered by the `c4v-py` library in a more efficient way for your environment or use case, or you maybe want access to a more robust way to perform queries. If that's the case, you can create your own persistency manager and configure the library to use it. 

!!!Info
    You can find a guide on **creating your own persistency manager** just right [here](../development/creating-a-persistency-manager.md)

## Environment variables
Assuming that you already have a persistency manager you want to add (you can use this [sample dict-based persistency manager](https://github.com/code-for-venezuela/c4v-py/blob/master/src/c4v/scraper/persistency_manager/example_dict_manager.py) for testing), all you have to do is to set up the right environment variables:

- Tell `c4v-py` that you want to use your own persistency manager:

```bash
export C4V_PERSISTENCY_MANAGER=USER
```

- Tell `c4v-py` the path to the file where to find the persistency manager.
```bash
export C4V_USER_PERSISTENCY_MANAGER_PATH=path/to/your/file.py
```

- Tell `c4v-py` the submodule of that file where to find the function to call in order to get a persistency manager. For example, if the function is in `file.py` as above, then this should be set:
```bash
export C4V_USER_PERSISTENCY_MANAGER_MODULE=file
```

Note that **this module must provide a function `get_persistency_manager`** that takes no args and returns an instance of the persistency manager object.

!!!Info 
    More on environment variables and configuration [here](configuration.md).

