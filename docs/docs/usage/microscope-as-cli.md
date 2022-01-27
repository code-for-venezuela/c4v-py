# Microscope as a CLI
The `c4v-py` library offers a CLI tool to perform a lot of operations. In this section, we will check some of the must useful features.

## Browsing stored data
The first thing you might want to see is how data is stored, you can do so by using the command:
```bash
c4v list
```

This command will **show up a table** in your terminal with some of the stored data, which will be... none the first time you run it. Let's try to get more data to fill this table!

```
row    author    categories    content    date    label_relevance    label_service    last_scraped    source    title    url
-----  --------  ------------  ---------  ------  -----------------  ---------------  --------------  --------  -------  -----
``` 

!!! Info
    The `list` command has many flags to help you filter how many and which data is shown to you. Check it using the `c4v list --help` command.

## Crawling
Let's say we want to add more urls to our database in order to see something in our database. To do so, we use the following command:

```bash
c4v crawl --limit 100 primicia
```

!!! Warning
    By default, the `crawl` command will try to crawl the entire site, which might be a bit expensive and take a long time, so we highly recommend you to set the `--limit` flag to a specific value, at the list during this examples

Let's see our new data with the `list` command:
```bash
c4v list
```

Now we might see a table with a urls only:

```
  row  author    categories    content    date    label_relevance    label_service    last_scraped    source    title    url
-----  --------  ------------  ---------  ------  -----------------  ---------------  --------------  --------  -------  -----------------------------------------------------
    1  None      []            None       None    None               None             None            None      None     https://primicia.com.ve/noticia1
    2  None      []            None       None    None               None             None            None      None     https://primicia.com.ve/noticia2
```

Now that we have some links in our database, we might want to know a bit more about them. We can do this by scraping them!

## Scraping
Scraping will search and store the data for every url stored that is not yet scraped. You can trigger a scraping process using the command:
```bash
c4v scrape
```

now we have data for every article, and we can see it using the `c4v list` command as before. But if we do so, we will realize that not every column was filled, the most interesting ones are the `label_relevance` and `label_service` columns. To fill those columns, we will need to use the `classify` command.

!!! Warning
    In order to use the `classify` command, you'll need:

    1. The `classification` installation profile. (More on installation profiles [here](installation.md))
    2. A trained model, as the library doesn't provides a model by itself. 

## Classification
The final goal the microscope project is to infer information about public services from the media. To achieve this, we can use the `classify` command that will tell you if a specified article is about public services or no.

You can check if it is relevant or not by using the command:
```bash
c4v classify branch_name/experiment_name www.url.to/your/scrapable/article
```

* The `branch_name` part tells the command in which branch to search for the model. Models are organized in branchs and experiments, so they're a bit easier to locate and sort between multiple experiments.
* The `experiment_name` part tells the command which experiment inside the specified branch holds the corresponding model to use.
* The final argument is a **scrapable** URL you want to check, it's not necessary to scrape it first if no available. 

This command, if successful, will print a message like this:
```
<article title> (www.url.to/your/scrapable/article)
                RelevanceClassificationLabels.IRRELEVANTE
                tensor([0.9914, 0.0086], grad_fn=<UnbindBackward>)
```

This message is telling us:

* Title
* Provided link
* Assigned Label
* A tensor with the score for each label

But, what if we want to classify the data we scraped before? We can do it by replacing the url argument with `pending`. This will tell the command that it wants to **classify every pending row**:

```
c4v classify pending
```

## And more!
There's a lot of features in the microscope library that are not covered here, you can always check the `c4v --help` command to find out about them