CREATE TABLE questions (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL
);

INSERT INTO questions (id, question) VALUES
  (1, 'What does the country promise as their 2030/2035 NDC target?'),
  (2, 'What years are these countries using as their baseline?'),
  (3, 'Are they reporting a business as usual (BAU) target rather than a base year target?'),
  (4, 'What sectors are covered by the target (Ex. Energy, Industrial Processes, Land use, land use change and forestry (LULUCF), etc.)?'),
  (5, 'What greenhouse gasses are covered by the target?'),
  (6, 'What are the emissions in the baseline year (if reported)?'),
  (7, 'What are the emissions levels under the BAU scenario if relevant (may require getting data from tables/graphs)?'),
  (8, 'What promises under this new version of the document are different from the previous version of their NDC?'),
  (9, 'What policies or strategies does the country propose to meet its targets?'),
  (10, 'Do they specify what sectors of their economy will be the hardest to reduce emissions in?');