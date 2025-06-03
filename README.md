# Bias Analysis Tool

A Flask-based web application for analyzing potential biases in datasets, with a focus on fairness metrics across protected attributes.

## Project Structure

```
.
├── app/
│   ├── __init__.py           # Flask application factory
│   ├── models/
│   │   └── bias_evaluator.py # Core bias analysis logic
│   ├── routes/
│   │   └── main.py          # Route handlers
│   ├── static/
│   │   └── css/
│   │       └── style.css    # Custom styles
│   ├── templates/
│   │   ├── base.html        # Base template with common elements
│   │   └── index.html       # Main application page
│   └── utils/
│       └── llm_utils.py     # LLM integration utilities
├── uploads/                  # Directory for uploaded files
├── NYPD_Complaint_Data_Historic_20250515_preprocessed.csv  # Sample dataset
├── preprocess_dataset.py     # Dataset preprocessing script
├── requirements.txt          # Python dependencies
└── run.py                    # Application entry point
```

## Key Components

### Core Application Files

- `run.py`: Entry point for the Flask application. Creates and runs the Flask app instance.
- `app/__init__.py`: Application factory that initializes Flask and registers blueprints.
- `app/routes/main.py`: Contains route handlers for file upload, analysis, and results display.

### Analysis Logic

- `app/models/bias_evaluator.py`: Implements the core bias analysis functionality:
  - Trains logistic regression models
  - Computes fairness metrics (precision, recall, F1-score)
  - Analyzes bias across protected attributes
  - Returns detailed metrics for overall and group-wise analysis

### Frontend

- `app/templates/base.html`: Base template with common HTML structure and styling
- `app/templates/index.html`: Main application interface with:
  - File upload form
  - Analysis parameters
  - Results display
  - Interactive visualizations
- `app/static/css/style.css`: Custom styling for the application

### Utilities

- `app/utils/llm_utils.py`: Handles LLM integration for:
  - Identifying protected attributes
  - Determining target variables
  - Analyzing column dependencies
- `preprocess_dataset.py`: Utility script for preprocessing large datasets:
  - Reduces dataset size
  - Handles data cleaning
  - Optimizes for analysis

### Data

- `NYPD_Complaint_Data_Historic_20250515_preprocessed.csv`: Preprocessed sample dataset containing:
  - Complaint information
  - Demographic data
  - Location details
  - Crime classifications

## Features

1. **Data Upload and Processing**
   - CSV file upload
   - Automatic data validation
   - File caching for improved performance

2. **Bias Analysis**
   - Protected attribute identification
   - Target variable determination
   - Correlation analysis
   - Fairness metrics calculation

3. **Results Visualization**
   - Overall performance metrics
   - Group-wise analysis
   - Interactive charts
   - Detailed metrics display

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python run.py
   ```

3. Access the application at `http://localhost:5000`

## Usage

1. Upload a CSV file through the web interface
2. Provide column descriptions
3. Set analysis parameters:
   - Number of rows to analyze
   - Test size
   - Maximum iterations
4. View the analysis results:
   - LLM recommendations
   - Protected attributes
   - Target variable
   - Excluded columns
   - Fairness metrics

## Dependencies

- Flask: Web framework
- pandas: Data manipulation
- scikit-learn: Machine learning and metrics
- numpy: Numerical computations
- Bootstrap: Frontend styling

## Notes

- The application uses a factory pattern for better testing and configuration
- File caching is implemented to improve performance with large datasets
- The preprocessed NYPD dataset is included for demonstration purposes
- Other datasets are excluded from the repository

# dsai2



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.ai.wu.ac.at/ikobiako/dsai2.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.ai.wu.ac.at/ikobiako/dsai2/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
