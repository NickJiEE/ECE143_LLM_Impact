# Impact of Large Language Models on UCSD Student Performance

## Overview
This project analyzes the effect of Large Language Models (LLMs) on UCSD student academic performance by examining whether the release of AI tools corresponds with measurable changes in grade distributions and course satisfaction metrics.

## Research Questions
- Do course grade distributions show significant shifts after major LLM release dates?
- How do post-LLM trends compare to long-term historical patterns?
- Are some departments (e.g., CSE, MATH, ECON) more affected than others?
- How has the introduction of LLMs impacted student course satisfaction?
- How does each LLM model update (e.g., ChatGPT 4.0 to 4.5) change the trend?

### Additional Research Topics
- Class sizes vs. enjoyment (ratings)
- Overall difficulty of different departments as students transition from lower to upper division coursework
- Quarter-based analysis (e.g., do students in winter perform better than spring?)

## Datasets

### Primary Dataset
**UCSD CAPEs Dataset (2007–2023)**
- Source: [Kaggle - UCSD CAPEs](https://www.kaggle.com/datasets/sanbornpnguyen/ucsdcapes)
- Size: ~63,000 entries
- Contents: Historical course evaluation data including:
  - Average GPA per course
  - Expected and received grades
  - Enrollment numbers
  - Instructor information
  - Department information

### Supplementary Datasets
- **UCSD Sunset**: [Community-sourced recent course data](https://sheeptester.github.io/ucsd-sunset/)
- **RateMyProfessors**: [Community professor ratings](https://www.ratemyprofessors.com/search/professors/1079?q=)

## Project Timeline

1. **Repository Setup & Toolkit Identification**
   - [x] Repository initialization
   - [ ] Select data analysis libraries and tools
   - [ ] Set up development environment

2. **Data Extraction & Cleaning (ETL Pipeline)**
   - [ ] Extract data from primary and supplementary sources
   - [ ] Clean and normalize data
   - [ ] Build ETL pipeline

3. **Historical Baseline Analysis**
   - [ ] Plot overall historical grade distribution trends
   - [ ] Establish pre-LLM baseline metrics

4. **LLM Release Date Mapping**
   - [ ] Identify major LLM release dates and events
   - [ ] Map release dates to grade distribution plots
   - [ ] Conduct temporal analysis

5. **Feature-Based Filtering & Analysis**
   - [ ] Filter by department
   - [ ] Filter by professor
   - [ ] Analyze upper vs. lower division patterns

6. **Sub-Hypothesis Exploration**
   - [ ] Cross-reference additional datasets
   - [ ] Investigate secondary research questions

7. **Testing, Documentation & CI/CD**
   - [ ] Write unit tests
   - [ ] Complete documentation
   - [ ] Set up continuous integration

## Repository Structure
```
[TODO: Add directory structure]
```

## Installation
```bash
[TODO: Add installation instructions]
```

## Usage
```bash
[TODO: Add usage examples]
```

## Dependencies
```
[TODO: List required Python packages and versions]
```

## Team Members
- Abhinit Saurabh
- Jeevan N V
- Matthew Merioles
- Minghong Sun
- Nick Ji

## Contributing
```
[TODO]
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- UCSD CAPEs data providers
- Community contributors to UCSD Sunset
- [TODO: other acknowledgments]
