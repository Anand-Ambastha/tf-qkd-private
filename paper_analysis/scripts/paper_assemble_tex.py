import os
TEXD = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')), 'paper_analysis', 'latex', 'paper_content') + '/'
pre = r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs,array,multirow}
\usepackage{caption}
\usepackage{needspace}
\usepackage[T1]{fontenc}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,width=0.95\linewidth}
\theoremstyle{plain}
\newtheorem{lemma}{Lemma}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\setlength{\tabcolsep}{4pt}
\title{Turbulence-averaged TF-QKD key rate:\\ result tables and analytical derivation}
\author{Content prepared from the repository \texttt{Anand-Ambastha/tf-qkd-private}}
\date{20 October 2026}
\begin{document}
\maketitle

\section{Scope and conventions}
This document collects (i) the tables that show why the log-normal model and the low-order analytic approximations cannot be used at long distance (Sec.~\ref{sec:fail}), (ii) the Gamma--Gamma results with and without beam wander (Sec.~\ref{sec:bw}), (iii) the analytic approximations of the Gamma--Gamma average and their errors, mean versus 2nd-order median versus 4th-order median (Sec.~\ref{sec:approx}), and (iv) the complete derivation from the Gamma--Gamma model to the cumulant--Euler formula and to the log-normal formula (Sec.~\ref{sec:derivation}). All formulas and numbers are those of the repository code (\texttt{src/}, \texttt{analytical\_validation/}, \texttt{results\_v2.3.0/}); the validation scripts were re-run and reproduce the stored result files (70 numerical checks pass, 0 fail).

\paragraph{Provenance.} Tables~\ref{tab:val}, \ref{tab:stress}, \ref{tab:delta}, \ref{tab:gg-single}, \ref{tab:expoint} are the repository result tables (\texttt{analytical\_validation/results/tables/}); Tables~\ref{tab:range-ln} and \ref{tab:bw-range} are copied from the repository reports. Tables~\ref{tab:lnvsgg}, \ref{tab:bw-skr}, \ref{tab:perarm-weak}, \ref{tab:perarm-mod} were computed for this document by calling the repository functions (\texttt{legacy\_baseline}, \texttt{cal\_derivatives}, \texttt{averaging}, \texttt{beam\_broadening}, the per-arm Gamma--Gamma channel) at the stated distances; the scripts are in \texttt{paper\_analysis/scripts/} of the repository.

\paragraph{Beam wander.} The repository has no stand-alone beam-wander model for the Gamma--Gamma channel. Beam wander enters through the turbulence-induced \emph{long-term beam broadening} $W_{\rm LT}$ (``GG+BB'', \texttt{gamma\_gamma\_bb}), which already contains the wander of the beam centroid; ``with beam wander'' in this document means GG+BB (Sec.~\ref{sec:bb}). Pointing jitter (\texttt{corrected\_gamma\_gamma}) is not used, to avoid counting the same effect twice.

\paragraph{Errors.} Every error is $100\,(\text{approximation}-\text{reference})/\text{reference}$ in percent. References are independent of the formulas under test: Gauss--Hermite/Simpson quadrature in $X=\ln(\eta/\eta_0)$ for the log-normal channel; tensor Gauss--Laguerre quadrature (single Gamma--Gamma channel; checked against Bessel-$K$ quadrature and Monte Carlo); numerical convolution of the Bessel-$K$ log-densities (per-arm Gamma--Gamma).

'''
params = open(TEXD + 'tables_params.tex').read()
sec2 = r'''
\section{Why the log-normal model and the low-order approximations fail at long distance}
\label{sec:fail}
Tables~\ref{tab:val}--\ref{tab:range-ln} give, in this order, the validity distance of the log-normal law, the disagreement between log-normal and Gamma--Gamma key rates, the loss of accuracy of the log-normal analytic formulas as the log-variance grows, the breakdown of the delta method, and the resulting maximum ranges.
''' + open(TEXD + 'tables_set1.tex').read() + r'''
\Needspace{8\baselineskip}\paragraph{Reading the tables.}
\begin{enumerate}
\item \emph{Validity.} The log-normal model ends at $\sigma_R^2(L/2)=1$, i.e.\ at $208.8$, $16.9$ and $1.37$~km for $C_n^2=10^{-17},10^{-15},10^{-13}$ (Table~\ref{tab:val}). The maximum ranges of Table~\ref{tab:range-ln} ($\approx290$~km) lie beyond these limits in all three classes.
\item \emph{Model error.} The log-normal scintillation index $e^{\sigma_R^2}-1$ diverges, while the Gamma--Gamma index saturates ($0.71$ at $\sigma_R^2=1$, $1.17$ at $4$). Already at $\sigma_R^2=1$ the log-normal key rate is $11.6$--$14.4\%$ below the Gamma--Gamma one, and $57$--$62\%$ below at $\sigma_R^2=4$ (moderate and strong class, Table~\ref{tab:lnvsgg}); the log-normal maximum range is $132.7$ and $13.1$~km against $293.2$ and $291.7$~km (Table~\ref{tab:range-ln}).
\item \emph{Analytic formulas inside the log-normal model.} The 4-term formula Eq.~\eqref{eq:ln} has an error of $-6\times10^{-6}\%$ at $\sigma_X^2=0.01$, $-0.14\%$ at $0.307$, $-3.0\%$ at $1.0$ and $-7.4\%$ at $1.5$; the 2-term formula $-1.5\%$, $-11.7\%$, $-21.8\%$ at $0.307$, $1$, $1.5$ (Table~\ref{tab:stress}). For $\sigma_X^2=2$--$3$ the largest and smallest of the three formulas differ by a factor $1.5$ ($\sigma_X^2=2$) and $2.4$ ($\sigma_X^2=3$).
\item \emph{Delta method.} The skewness of $\eta$ is $\gamma_1=41$ at $\sigma_X^2=2.4$ and the mean-based expansion errs by $+28.8\%$; in $\eta_{\rm tot}$ it errs by $-20.1\%$ already at $\sigma_X^2=0.31$ (Table~\ref{tab:delta}). Caution: at a given operating point the delta method can be accurate far beyond this (Table~\ref{tab:stress}: within $1.3\%$ up to $\sigma_X^2=1.5$ at $L=80$~km), because $R(\eta)$ is locally close to a power law $\eta^{p}$ with $p=\Theta R/R=1.36,\,1.05,\,1.13$ at $L=20,80,160$~km (derivatives of the repository), so the curvature term is small. Its accuracy therefore depends on the operating point rather than on a controlled expansion parameter.
\end{enumerate}
'''
sec3 = r'''
\section{Gamma--Gamma results with and without beam wander}
\label{sec:bw}
Per-arm Gamma--Gamma fading (independent arms of length $L/2$, $\eta=\eta_A\eta_B$) without and with turbulence-induced long-term beam broadening, which contains beam wander. Broadening is a deterministic loss on the nominal transmittance of each arm, $\eta_{\rm arm}\to\eta_{\rm arm}f_{\rm bb}$ with $f_{\rm bb}=[1-e^{-2a^2/W_{\rm LT}^2}]/[1-e^{-2a^2/W^2}]$ and $W_{\rm LT}^2=W^2(1+1.63\,\sigma_R^{12/5}\Lambda)$ (Sec.~\ref{sec:bb}); the fading law and $(\alpha,\beta)$ are unchanged.
''' + open(TEXD + 'tables_set2.tex').read() + r'''
\Needspace{8\baselineskip}\paragraph{Reading the tables.}
\begin{enumerate}
\item \emph{Weak class.} Beam wander costs at most $6\%$ of the key rate at $L=160$~km ($R_{\rm GG+BB}/R_{\rm GG}=0.94$) and $3.8$~km ($1.3\%$) of range (Table~\ref{tab:bw-range}).
\item \emph{Moderate class.} The ratio falls to $0.26$ at $50$~km, $0.14$ at $80$~km and $0.018$ at $160$~km; the range shrinks by $94.6$~km ($32.3\%$).
\item \emph{Strong class.} The long-term beam is $1.7$~m at $20$~km and $15.7$~m at $80$~km ($W=0.11$ and $0.40$~m), the rate is below $10^{-9}$ from $\approx80$~km on, and the range drops by $209.8$~km ($71.9\%$). Plain Gamma--Gamma, which only redistributes power around a fixed mean, keeps a range of $\approx292$~km for all three classes.
\item \emph{Assumptions to state when quoting.} Horizontal path, constant $C_n^2$, collimated Gaussian transmitter; $\alpha,\beta$ keep their plane-wave values (first-order correction); no extra pointing jitter on top of $W_{\rm LT}$; the broadening law keeps growing for $\sigma_R^2\gg1$, so strong-class GG+BB values are lower bounds.
\end{enumerate}
'''
sec4 = r'''
\section{Analytic approximations of the Gamma--Gamma average and their errors}
\label{sec:approx}
Three approximations are compared, all evaluated with the analytic derivatives of the repository: \emph{mean} (delta method about $\mathbb E[\eta]$ or $\mathbb E[\eta_{\rm tot}]$), \emph{2nd-order median} and \emph{4th-order median} (Eq.~\eqref{eq:main} truncated after $\Theta^2$ resp.\ $\Theta^4$, expanded about the log-mean point $\eta_m=\eta_0e^{c_1}$). Table~\ref{tab:gg-single} is the progress-report setting (one cumulative channel, aperture-averaged); Tables~\ref{tab:perarm-weak} and \ref{tab:perarm-mod} use the per-arm model of the code, without and with beam wander.
''' + open(TEXD + 'tables_set3.tex').read() + r'''
\Needspace{8\baselineskip}\paragraph{Reading the tables.}
\begin{enumerate}
\item \emph{Single channel with aperture (Table~\ref{tab:gg-single}).} For $\sigma_R^2(L)$ between $0.05$ and $5$ the 4th-order median formula is within $0.023\%$ of quadrature, the 2nd order within $0.084\%$, the 3rd order within $0.31\%$ and the delta method within $1.2\%$; the log-normal formula applied to the same channel errs by $-1.3\%$ at $\sigma_R^2=0.26$ and $-52.6\%$ at $5$. The 3rd order is worse than the 2nd (Sec.~\ref{sec:order}). The median (log-mean) point beats the mean and the mode in every row (Table~\ref{tab:expoint}).
\item \emph{Per arm, weak class, with and without beam wander (Table~\ref{tab:perarm-weak}).} For $\sigma_R^2(L/2)\le0.61$ the 4th-order median error is $\le0.005\%$ for $e_X$, $\le0.0014\%$ for $e_Z$ and $\le0.042\%$ for $R$ ($L=160$~km); mean and 2nd order give $-0.14\%/-0.066\%$ ($e_X$), $-0.082\%/-0.036\%$ ($e_Z$) and $+0.055\%/-0.30\%$ ($R$). Beam wander changes these errors by less than $0.01$ percentage points, because it only shifts $\eta_0\to\eta_0f_{\rm bb}$ ($f_{\rm bb}^2\ge0.90$). For $R$ the delta method is competitive with the 4th order (see the power-law remark in Sec.~\ref{sec:fail}); for $e_X$ and $e_Z$ the 4th order is one to two orders better.
\item \emph{Per arm, moderate class (Table~\ref{tab:perarm-mod}).} With $\beta\to1$ the log-variance $c_2$ is $O(1)$ and the expansion is no longer controlled: the 4th order errs by $+12.6\%$ at $20$~km and $+2.0$ to $+2.7\%$ for $L=50$--$160$~km, better than the 2nd order only at $50$~km and worse for $L\ge80$~km; with beam wander at $160$~km ($f_{\rm bb}^2=4.5\times10^{-3}$) the expansion point lies below the key-rate cut-off and all approximations fail ($-41\%$ to $-51\%$ for $R$). The formula is therefore reliable for $c_2\lesssim0.3$ (weak class and aperture-averaged channels), not in saturated moderate/strong turbulence.
\end{enumerate}
'''
der = open(TEXD + 'derivation.tex').read()
sec6 = r'''
\section{Notes on the sources, and reproducibility}
\label{sec:notes}
\paragraph{Discrepancies found while reproducing the printed numbers.}
\begin{enumerate}
\item \emph{Gamma--Gamma exponent.} The progress report's Gamma--Gamma figures and tables are reproduced only with $\sigma_R^{12/5}$ evaluated as $(\sigma_R^2)^{12/5}$ (scintillation index peak $0.71$, falling to $\approx0.08$ at $\sigma_R^2=10$). In Andrews--Phillips $\sigma_R$ is the Rytov standard deviation, i.e.\ the exponent is $6/5$ on the variance; then $\mathrm{SI}$ peaks at $\approx1.2$--$1.3$ near $\sigma_R^2=5$--$10$ and saturates towards $1$. The repository code has used the correct exponent since v2.2.0. The cumulant formulas and the accuracy of the 4th-order formula are unaffected (Table~\ref{tab:gg-single}), but 26 of the 40 printed Table~1 entries differ by more than $0.25$ percentage points under the correct exponent, concentrated in the aperture-gain and delta-method columns at $\sigma_R^2>1$.
\item \emph{$L=20$~km row of the progress report.} It prints $+0.36\%$ in the delta, 2nd, 3rd and 4th-order columns; a formula-independent offset points to the reference integral at very large shapes ($\alpha,\beta\approx43,162$), not to the expansion.
\item \emph{Manuscript, Eq.~(6)/Appendix~A.} The $A^2$ term of $p_{XX}$ sits outside the factor $\tfrac12$: $p_{XX}=\tfrac A2[e^{-(1-\Omega)\gamma}+e^{-(1+\Omega)\gamma}]-A^2e^{-2\gamma}$ (as in CAL Eq.~(30) and the code). The printed bracket gives $p_{XX}=0.5$ instead of $1.57\times10^{-3}$ at $L=80$~km.
\item \emph{Appendix~E.} $D_j$ does not depend on $\eta$ in CAL and in the code, so $D_j^{(n)}=0$ for $n\ge1$; the statement that $D_j^{(n)}$ follows from the $\sqrt{p_{ZZ}}$ terms holds only for a different $D_j$.
\item \emph{Appendix~H.} ``$\sigma_X^2=9.6$'' for $C_n^2=10^{-16}$, $L=140$~km should read $2.4$ (it equals $4\sigma_X^2$); the skewness $41$ and the error $+29\%$ are reproduced.
\item \emph{Naive key rate becoming negative.} In the single cumulative log-normal channel of the report the delta-method key rate first reaches $\le0$ at $\sigma_R^2\approx2.3$ (``near two''); in the two-arm convention of the manuscript it does not become negative up to $\sigma_X^2=3$.
\end{enumerate}
\paragraph{Reproducibility.} The repository results are regenerated by \texttt{python3 analytical\_validation/run\_all.py} (about $30$~s; needs \texttt{mpmath}). Tables~\ref{tab:bw-skr}, \ref{tab:lnvsgg}, \ref{tab:perarm-weak}, \ref{tab:perarm-mod} come from \texttt{paper\_analysis/scripts/gg\_beamwander\_table.py}, \texttt{ln\_vs\_gg\_per\_arm.py} and \texttt{perarm\_analytic\_errors.py} (outputs in \texttt{paper\_analysis/results/tables/}), which import the repository unchanged.

\begin{thebibliography}{9}
\bibitem{AP} L.~C.~Andrews and R.~L.~Phillips, \emph{Laser Beam Propagation through Random Media}, 2nd ed., SPIE Press (2005).
\bibitem{AAP} M.~A.~Al-Habash, L.~C.~Andrews, R.~L.~Phillips, Opt.\ Eng.\ \textbf{40}, 1554 (2001).
\bibitem{CAL} M.~Curty, K.~Azuma, H.-K.~Lo, npj Quantum Inf.\ \textbf{5}, 64 (2019); arXiv:1807.07667.
\end{thebibliography}
\end{document}
'''
open(TEXD + 'tfqkd_paper_content.tex', 'w').write(pre + params + sec2 + sec3 + sec4 + der + sec6)
