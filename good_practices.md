# Good practices in reproducibility for computational researches

When we think about reproducibility in computational researches, we are not simple
talking about share the data, the code and a paper. We must figure out about a
lot of  problems that could block other researchers to reproduce our results.

This problems could be organized in five main elements of reproducibility, they are:

## Data

Think how share the data is really hard. You must consider how large is your data,
look for the data license and terms of use, check if the data has any ethical issue.
Moreover, you must think in turn available the dataset you use not only the database.
Also, the platform used to share the data is really important, because we need
the data to be consistent and the link to it must be persistent. Some tools offer
data versioning.

Another think to be worry is about data manipulation. Avoid all kind of manual data
manipulation. If you can make scripts (shell, python, ...) to automatize your data
handling. If you can not, document every movement on your data, or in the software
used to do that.

## Code

Version the code and programming using good practices are basic actions to do
think or not in reproducibility. Together with this is import to have a good
code documentation. Documentation is not about putting comments in every part
of your code or creating simple README.md files. Document a code is more about
teaching others how to build the code, use it, describe the logic behind it and
obtain the same results as you.

## Documentation

Besides the code documentation, a full documented computational research will
describe details about the research area, the methodology used, the experiments
done, and all things related to the research (results, charts, paper, code to
generate results, ...). A tip here is use executable papers such as Jupyter
Notebook, Google Colab, Polynote, Next Journal, and others.

## Workflow

Describe your global workflow is important to others researches understand how
the modules of code/data/results are integrated.

However, a local workflow documentation are very important for reproducibility.
Others researches must now the data manipulation actions, the chosen dataset,
the experiment done, the parameters of this experiment, the random seed used and
other thinks in other to obtain exactly the same results as you.

Tracking experiments and its parameters could also help you to control your
research results. Some tools does it to you, avoiding you to have to update
a spreadsheet with parameters and results after each experiment execution.
Some of this tools are: CodaLab, Reana, MlFlow, Comet, Sacred, ...

## Environment and Infrastructure

A crucial element to ensure reproducibility is share or detailed document the
research computational environment. To do that pay attention in:

- report the operational system used;
- ensure that all packages have their versions reported;
- external links, since they could become unavailable, so give all kind of
  description that could help others find the external program/data used.

During environment sharing really pay attention in packages versions. To do this
you could use Docker, Virtual Machines, Anaconda, or other tools. Try to have more
than one option, and think in different operational systems users.

About infrastructure think that not everyone has a server or GPU to execute your
code exactly as you do. To avoid this create a simple version of your could that
could run in simple CPUs. If not possible, place visible warnings in your documentation.

In documentation do not forget to teach how to build the environment, and check all
the external links.

## Final Tip

Try yourself to reproduce your research using a totally new environment.