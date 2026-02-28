\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ems-journal}[2025/02/28 EMS Journal Fake Class]
\LoadClass{article}
\RequirePackage{amsthm,amsmath,amssymb,amsfonts}
\RequirePackage{enumitem}
\RequirePackage{graphicx}
\RequirePackage{hyperref}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\newcommand{\emsauthor}[3]{\author{#1}}
\newcommand{\givenname}[1]{#1}
\newcommand{\surname}[1]{#1}
\newcommand{\orcid}[1]{}
\newcommand{\Emsaffil}[2]{}
\newcommand{\department}[1]{}
\newcommand{\organisation}[1]{}
\newcommand{\rorid}[1]{}
\newcommand{\address}[1]{}
\newcommand{\zip}[1]{}
\newcommand{\city}[1]{}
\newcommand{\country}[1]{}
\newcommand{\affemail}[1]{}
\newcommand{\classification}[2][]{}
\newcommand{\keywords}[1]{\textbf{Keywords:} #1}
\newcommand{\titlemark}[1]{}
\newenvironment{ack}{\section*{Acknowledgments}}{}
\newenvironment{funding}{\section*{Funding}}{}
