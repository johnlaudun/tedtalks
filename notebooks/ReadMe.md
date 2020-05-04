# ReadMe.md

A Notebook file is created for each significant piece of experimentation, with a relevant title and a versioning number for each time the experiment is repeated using different parameters or a new way of doing things.

While we are working, we tend to keep all our notebooks in one big pile, sorting by modification date and time in Jupyter's file explorer view. As Serban notes: "The advantage of keeping things together is that you can implement cross referencing within and between notebooks. This is simply done with html links in Markdown."

To create a reference to a section of your Notebook, add the following code in a Markdown cell before the referenced part:

```html
<a id='label_of_your_choice'></a>
<!--referenced section in file experiment00.ipynb-->
```

Now, in the place where you want your link to the reference to appear write one of the following:

```markdown
[description](#label_of_your_choice) <!--to make link in the same notebook-->
[description](experiment00.ipynb#label_of_your_choice) <!--to make link in another notebook-->
````


[How to organize code in Python if you are a scientist](https://towardsdatascience.com/workflow-for-reportable-reusable-and-reproducible-computational-research-45d036c8a908)