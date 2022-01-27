# Installation
You can install the microscope library by just using the command:

```bash
pip install c4v-py
```

This command will include only the scraping features of `c4v-py`. If you want to include more features, you might consider a different installation profile.

## Installation profiles
To install an alternatvie profile, all you have to do is to use this command:
```bash
pip install c4v-py[<profile name>]
```

where `<profile name>` could be one of the following:

- `classification` : Support for classification, allowing you to use classifiers and experiments.
- `tensorflow` : Include tensorflow as dependency.
- `dashboard` : Include the dashboard to perform common operations from a web browser.
- `jupyter` : Include jupyter as dependency.
- `gcloud` : Support for cloud features, such as using a cloud backend in the dashboard, or loading and downloading classifiers from the cloud.
- `all` : As you might expect, includes **every dependency** in all profiles.
