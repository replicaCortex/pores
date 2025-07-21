\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{wordcore}[2025/07/07 Custom Word-like LaTeX class]

% — Base class —
\LoadClass[14pt]{extarticle}

% — Required packages —
\RequirePackage{cmap}
\RequirePackage{indentfirst}       % отступ в начале первого абзаца
\RequirePackage{geometry}          % настройки полей страницы
\RequirePackage{caption}           % стили подписей
\RequirePackage{float}             % позиционирование float-объектов
\RequirePackage{booktabs}          % книжный вариант таблиц 
\RequirePackage{fancyhdr}          % footnotes
\RequirePackage{graphicx}          % вставка изображения
\RequirePackage{subcaption}
\RequirePackage[russian]{babel}
\RequirePackage{microtype}
\RequirePackage{titlesec}
\RequirePackage{tocloft}

\RequirePackage[
	backend=biber,
	style=gost-numeric,    % или любой другой стиль
]{biblatex}
\addbibresource{bib.bib}

\newcommand{\bbodysection}[1]
{
	\begingroup
	\titleformat{\section}[block]
	{\normalfont\large\bfseries\centering}
	{\thesection}{1em}{}

	\section*{#1}

	% \stepcounter{section}
	\addcontentsline{toc}{section}{#1}
	\endgroup
}


\newcommand{\pprintbibliography}
{
	\bbodysection{СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ}
	\printbibliography[heading=none]
}

\newcommand{\ttableofcontents}{
	\begingroup
	\titleformat{\section}[block]
	{\normalfont\large\bfseries\centering}
	{\thesection}{}{}

	\section*{СОДЕРЖАНИЕ}
	\endgroup
	\renewcommand{\contentsname}{\vspace{-1cm}}
	\tableofcontents
}

% — Settings —   
\newif\if@listing
\DeclareOption{code}{\@listingtrue}

\newif\if@section
\DeclareOption{section}{\@sectiontrue}

\ProcessOptions\relax

% — Minted —   
\if@listing
	\RequirePackage[newfloat]{minted}    % подсветка кода
	\setminted{
		breaklines,              % переносить строки
		fontsize=\footnotesize,  % размер шрифта
		tabsize=2,               % табуляция
		style=bw                 % Pygments тема
	}

	\DeclareCaptionLabelFormat{ruListing}{Листинг #2}
	\captionsetup[listing]{labelformat=ruListing, labelsep=endash}

	\newenvironment{code}
	{\par\captionsetup{type=listing}\noindent}
	{\par}
	\captionsetup[listing]{labelformat=ruListing, labelsep=endash}
	% \else
	% \PackageWarning{wordcore}{[code] not load}
\fi

% — Section formatting —

\if@section

	\makeatletter
	\renewcommand{\section}{\@startsection
		{section}{0}{0em}
		{-\baselineskip}{0.5\baselineskip}
		{\bfseries\LARGE}
	}
	\renewcommand{\subsection}{\@startsection
		{subsection}{1}{0em}
		{-\baselineskip}{0.5\baselineskip}
		{\bfseries\Large}
	}
	\renewcommand{\subsubsection}{\@startsection
		{subsubsection}{2}{0em}
		{-\baselineskip}{0.5\baselineskip}
		{\bfseries\Large}
	}
	\renewcommand{\paragraph}{\@startsection
		{paragraph}{3}{0em}
		{-\baselineskip}{0.5\baselineskip}
		{\normalfont\normalsize}
	}
	\renewcommand{\subparagraph}{\@startsection
		{subparagraph}{4}{0.7cm}
		{-\baselineskip}{0.5\baselineskip}
		{\normalfont\normalsize}
	}
	\makeatother
	% \else
	% 	\PackageWarning{wordcore}{[section] not load}
\fi{}

% — Geometry —
\geometry{
	left=3cm,
	right=1.5cm,
	top=2cm,
	bottom=2cm,
	includefoot
}

% — Caption settings —
\DeclareCaptionLabelFormat{ruFigure}{Рисунок #2}
\captionsetup[figure]{labelformat=ruFigure, labelsep=endash}

\DeclareCaptionLabelFormat{ruTable}{Таблица #2}
\captionsetup[table]{labelformat=ruTable, labelsep=endash, justification=raggedright, singlelinecheck=false}


% — Paragraph settings —
\RequirePackage{setspace}
\onehalfspacing
\setlength{\parindent}{1.25cm}
% \sloppy                               % избегать переполнений строки


\RequirePackage{enumitem}

% ——————————————————————————————————————————————————
% Глобальные настройки списков
\setlist[itemize]{%
	label=–,             % тире как маркер
	leftmargin=\parindent,
	itemsep=\parskip,
	topsep=\parskip
}

\setlist[enumerate]{%
	label=\arabic*),     % формат нумерации: 1), 2), ...
	leftmargin=\parindent,
	itemsep=\parskip
}
% ——————————————————————————————————————————————————

% \RequirePackage{titlesec} % Управление стилями заголовков

% % ——————————————————————————————————————————————————
% Глобальное отключение нумерации заголовков
% \titleformat{\section}[block]{\normalfont\Large\bfseries}{}{0pt}{}
% \titleformat{\subsection}[block]{\normalfont\large\bfseries}{}{0pt}{}
% \titleformat{\subsubsection}[block]{\normalfont\normalsize\bfseries}{}{0pt}{}
% \titleformat{\paragraph}[runin]{\normalfont\normalsize\bfseries}{}{0pt}{}
% \titleformat{\subparagraph}[runin]{\normalfont\normalsize\bfseries}{}{0pt}{}
%
% % Чтобы не было пробелов для нумерации (если включена)
% \setcounter{secnumdepth}{0}


\titleformat{\section}        % форматируем \subsection
{\normalfont\large\bfseries}       % стиль: обычный шрифт, обычный размер
{ГЛАВА \thesection. }               % номер (например, 1.1)
{0pt}                          % отступ между номером и текстом
{}                             % перед заголовком — ничего

\titleformat{\subsection}        % форматируем \subsection
{\normalfont\normalsize}       % стиль: обычный шрифт, обычный размер
{\thesubsection}               % номер (например, 1.1)
{1em}                          % отступ между номером и текстом
{}                             % перед заголовком — ничего

\titleformat{\subsubsection}        % форматируем \subsection
{\normalfont\normalsize}       % стиль: обычный шрифт, обычный размер
{\thesubsubsection}               % номер (например, 1.1)
{1em}                          % отступ между номером и текстом
{}                             % перед заголовком — ничего

\RequirePackage{tocloft}

% Подключаем точки в оглавлении
\renewcommand{\cftdot}{.}          % символ точки
\renewcommand{\cftsecleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftsubsecleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftsubsubsecleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftsecpresnum}{ГЛАВА~}
\renewcommand{\cftsecaftersnum}{.}
\setlength{\cftbeforesecskip}{5pt}

% Настройка выравнивания номеров
\cftsetindents{section}{0em}{2.0em}
\cftsetindents{subsection}{2.0em}{2em}
\cftsetindents{subsubsection}{4.0em}{2.7em}

\RequirePackage{amsmath}
\numberwithin{table}{section}
\numberwithin{figure}{section}

\RequirePackage{csquotes}

\renewcommand{\texttt}[1]{{\small\ttfamily #1}}

\endinput
