# Mathematical model specification

The continental calculations use a retained crop-development and STB formulation. Model labels do not establish measured anatomy or physiological loss. Rolling monitoring adds a separate observation-conditioned statistical correction; its inputs differ from the four-weather-variable projection interface. Numerical parameter identities are retained in parameters/ and Tables S4–S5.

### Calendar, state boundary and developmental clocks

The daily forcing comprises mean temperature \(T_d\) and maximum temperature \(T_d^{\max}\) in °C, mean relative humidity \(R_d\) in percent, and precipitation \(P_d\) in mm. The crop indicator \(c_d\) equals one from the supplied sowing date through the supplied crop-end date, inclusive. Regional seasons end at the first predicted terminal-stage crossing. The state indexed by \(d\) is the state at the end of day \(d\); initial developmental clocks are zero. Weather outside the crop window advances the calendar while freezing developmental clocks, tissue and local source potential. The forcing is consecutive, and a fresh simulation includes the sowing date. Accepted daily mean temperatures are at most 40 °C, the upper support of the retained thermal response.

Let \(\operatorname{clip}(x,a,b)=\min\{b,\max(a,x)\}\). Daily effective thermal accumulation is

\[
g_d=c_d\begin{cases}
\operatorname{clip}(T_d,0,20),&T_d\leq30,\\
20-2(T_d-30),&30<T_d\leq40,
\end{cases}
\qquad G_d=G_{d-1}+g_d. \tag{S1}
\]

For latitude \(\phi\) in radians and calendar day of year \(j_d\), the retained astronomical and photoperiod functions are

\[
\delta_d=0.4093\sin\{2\pi(j_d-81)/365\},\qquad
\ell_d={24\over\pi}\arccos\!\left[\operatorname{clip}\{-\tan\phi\tan\delta_d,-1,1\}\right],
\qquad p_d=\operatorname{clip}\{1-0.09(16-\ell_d),0,1\}. \tag{S2}
\]

Photoperiod sensitivity depends on the prior-day thermal clock:

\[
b_d^P=c_d\mathbf1(G_P^{\rm on}\leq G_{d-1}<G_P^{\rm off}),\qquad
u_d=g_d\{1-b_d^P+b_d^Pp_d\},\qquad U_d=U_{d-1}+u_d. \tag{S3}
\]

The temperature response for vernalization is

\[
v(T)=\begin{cases}
0,&T\leq-4\ \text{or}\ T\geq16,\\
(T+4)/4,&-4<T<0,\\
1,&0\leq T\leq10,\\
(16-T)/6,&10<T<16.
\end{cases} \tag{S4}
\]

For vernalization-required cases, define \(b_d^V=c_d\mathbf1(U_{d-1}\geq U_V^{\rm on})\), \(s_d^V=\mathbf1(U_{d-1}<U_V^{\rm off})\), and \(\widetilde V_d=V_{d-1}+s_d^Vv(T_d)\). The candidate and committed vernalization states are

\[
V_d^*=\begin{cases}
\max\{0,V_{d-1}-0.5(T_d^{\max}-30)\},&\widetilde V_d<10\ \text{and}\ T_d^{\max}>30,\\
\widetilde V_d,&\text{otherwise},
\end{cases}
\qquad V_d=(1-b_d^V)V_{d-1}+b_d^VV_d^*, \tag{S5}
\]

\[
f_d^V=\begin{cases}0.3+0.7\min(V_d/40,1),&b_d^Vs_d^V=1,\\1,&\text{otherwise},\end{cases}
\qquad A_d=A_{d-1}+u_df_d^V. \tag{S6}
\]

Without a vernalization requirement, \(V_d=V_{d-1}\), \(f_d^V=1\) and \(A_d=U_d\). Photoperiod and vernalization gates use \(G_{d-1}\) and \(U_{d-1}\), whereas the vernalization response multiplier uses the updated \(V_d\). Thus current weather affects current development without an observed stage entering the calculation. \(G\), \(U\) and \(A\) have effective °C d units; \(V\) has effective-day units. Effective TPV units are a developmental index and do not establish a measured leaf phyllochron. The retained development multiplier is \(q=1\).

For a supplied developmental threshold \(C_k\), the predicted stage date is

\[
\tau_k=\min\{d:c_d=1,\ A_d\geq C_k\}. \tag{S7}
\]

An unreached stage remains missing. BBCH codes are categorical event labels; numerical BBCH interpolation is absent. Additional thresholds \(C_{32},C_{33},C_{37},C_{39}\) are ordered stage-only estimates. The flowering proxy satisfies \(C_{65}=C_{51}+0.2317(C_{85}-C_{51})=1165.925558\) effective °C d. The profile tolerance sets for these estimates describe calibration-loss compatibility, rather than confidence intervals.

### Final-leaf capacity and juvenile handover

Slots \(i=0,\ldots,6\) denote F1–F7, with F1 the flag leaf; slot \(i=7\) is an aggregate juvenile canopy. Final-leaf visibility and full-unfolding thresholds are

\[
B_i=\max\{C_{10},C_{37}-i\Delta\},\qquad E_i=B_i+(C_{39}-C_{37}), \tag{S8}
\]

where the selected effective rank spacing is \(\Delta=120\) TPV units. End-day visibility is \(c_d\mathbf1(A_d\geq B_i)\), and full unfolding is \(c_d\mathbf1(A_d\geq E_i)\). Availability for disease requires prior-day visibility:

\[
H_{i,d}=c_dc_{d-1}\mathbf1(A_{d-1}\geq B_i),\qquad
a_{i,d}=H_{i,d}\operatorname{clip}\left({A_d-B_i\over E_i-B_i},0,1\right),\qquad
h_{i,d}=\mathbf1(a_{i,d}>0). \tag{S9}
\]

The initial prior crop indicator is zero. Visibility on the first threshold-crossing day therefore does not immediately enable infection. Daily susceptible renewal associated with final-leaf expansion is

\[
\rho_{i,d}=\begin{cases}\max(a_{i,d}-a_{i,d-1},0)/a_{i,d},&a_{i,d}>0,\\0,&a_{i,d}=0.\end{cases} \tag{S10}
\]

The primary juvenile policy gives

\[
J_d=\operatorname{clip}\left({C_{39}-A_d\over C_{39}-C_{31}},0,1\right),\qquad
a_{7,d}=c_dc_{d-1}\mathbf1(A_{d-1}\geq C_{10})J_d,\qquad
\rho_{7,d}=\mathbf1(a_{7,d}>0)\{1-\exp[-\max(T_d,0)/300]\}. \tag{S11}
\]

The alternative persistent-juvenile scenario sets \(J_d=1\). Juvenile capacity withdrawal is not leaf unfolding and does not activate contact reception. Capacities \(a_{i,d}\) are normalized and dimensionless. Natural senescence, absolute green area and true final main-stem leaf number are absent from this capacity formulation. Lower ranks clipped at \(C_{10}\) have unresolved shared appearance thresholds.

### Normalized tissue, effective sources and daily numerical transitions

Each slot contains susceptible tissue \(S_i\), \(m\) latent compartments \(L_{i,1},\ldots,L_{i,m}\), visibly affected noninfectious tissue \(N_i\), infectious tissue \(I_i\), and retained damage \(D_i\):

\[
S_i+\sum_{r=1}^{m}L_{i,r}+N_i+I_i+D_i=1. \tag{S12}
\]

All tissue states are normalized fractions. Initial tissue is entirely susceptible. Local source potential has relative unready and competent states \(Q^u,Q^c\), initialized as \(((1-f_0)Q_0,f_0Q_0)\), with \(Q_0=1\) and \(f_0=0.5\). These states are aggregate source phenomenology; their units are arbitrary relative-source units, with no measured-spore or reproductive-maturation interpretation.

At the beginning of each included crop day, inactive slots reset to \((1,0,\ldots,0)\). In active slots, all nonsusceptible compartments are multiplied by \(1-\rho_{i,d}\), and \(S_i\) becomes one minus their sum. This operation represents capacity dilution or juvenile renewal. Outside the crop window the complete tissue and source states freeze, with no inactive-slot reset.

One day contains \(n=\lceil1/\Delta t_{\rm req}\rceil\) equal substeps of length \(\eta=1/n\) d. The frozen setting \(\Delta t_{\rm req}=0.25\) d gives \(n=4\). Weather and host capacity remain fixed within the day. Define \(\theta_d=\max(T_d,0)/18\), \(r_d=1-\exp(-P_d/p_*)\), with \(p_*=2\) mm. At the beginning of each substep, area-weighted infectious neighborhood and adjacency are

\[
Z_{i,d}={\sum_{j:h_{j,d}=1}\exp(-|i-j|/\lambda)I_j a_{j,d}\over\sum_{j:h_{j,d}=1}a_{j,d}},\qquad
C_{i,d}={\sum_{j:h_{j,d}=1,\ |i-j|=1}I_j a_{j,d}\over\sum_{j:h_{j,d}=1}a_{j,d}}, \tag{S13}
\]

with both quantities zero if the denominator is zero. Here \(\lambda=2\) rank intervals. The four effective establishment hazards are

\[
\begin{aligned}
z_i^{\rm local}&=\alpha e_dQ^c\{(1-\omega)r_d\exp[-(7-i)/\lambda]+\omega\},\\
z_i^{\rm imported}&=\alpha e_d M_d,\\
z_i^{\rm splash}&=\beta e_dr_d Z_{i,d},\\
z_i^{\rm contact}&=\beta e_d\kappa(1-a_{i,d})C_{i,d}\mathbf1(i\ne7).
\end{aligned} \tag{S14}
\]

\(e_d\) is dimensionless weather exposure, \(M_d\) is supplied relative imported pressure, \(\omega=0.1\), and \(\kappa=0.2\). The selected primary and secondary coefficients are \(\alpha=0.001\) and \(\beta=10\); primary normalization is per relative-source unit per day, and secondary normalization is per day. Their scales depend on the assumed source normalization and tissue capacity. The rank kernel includes the receiving slot; imported pressure does not require local source or precipitation.

For \(z_i=\sum_pz_i^p\), newly infected tissue and pathway increments are

\[
F_i=S_i\{1-\exp(-\eta z_i)\},\qquad
F_i^p=\begin{cases}F_i z_i^p/z_i,&z_i>0,\\0,&z_i=0.\end{cases} \tag{S15}
\]

Only active slots receive these updates. Pathway outputs sum substep increments over the day; they are newly affected fractions, without a spore-count or identified causal-attribution interpretation. The thermal transition probabilities are

\[
q_L=1-\exp(-\eta m\theta_d/\tau_L),\qquad
q_N=1-\exp(-\eta\theta_d/\tau_N),\qquad
q_I=1-\exp(-\eta\theta_d/\tau_I). \tag{S16}
\]

With pre-substep states on every right-hand side, the simultaneous update is

\[
\begin{aligned}
S_i'&=S_i-F_i,\\
L_{i,1}'&=(1-q_L)L_{i,1}+F_i,\\
L_{i,r}'&=(1-q_L)L_{i,r}+q_LL_{i,r-1},\quad r=2,\ldots,m,\\
N_i'&=(1-q_N)N_i+q_LL_{i,m},\\
I_i'&=(1-q_I)I_i+q_NN_i,\\
D_i'&=D_i+q_II_i.
\end{aligned} \tag{S17}
\]

New infections do not progress through a latent stage in their substep of entry. Reference durations \(\tau_L=20\), \(\tau_N=3\) and \(\tau_I=21\) d refer to the 18 °C clock; \(m=3\). The latent chain has a continuous-time reference mean of \(\tau_L\), while realized residence times follow the explicit finite-substep updates.

For source maturation and common ageing rates \(\mu_d=\theta_de_d/\tau_M\) and \(\nu_d=\theta_d/\tau_Q\), with \(\tau_M=30\) and \(\tau_Q=90\) reference d, the implemented source update is

\[
(Q^u)'=Q^u\exp[-\eta(\mu_d+\nu_d)],\qquad
(Q^c)'=\exp(-\eta\nu_d)\{Q^c+Q^u[1-\exp(-\eta\mu_d)]\}. \tag{S18}
\]

Consequently \((Q^u+Q^c)'=(Q^u+Q^c)\exp(-\eta\nu_d)\) exactly. Tissue and source proposals use their beginning-of-substep states and are committed together. Nonpositive mean temperature stops the thermal progression and source-ageing clocks; it does not force weather exposure or establishment hazards to zero. Renewal and availability also affect tissue independently of the thermal transition probabilities.

### Weather exposure and observation operator

For hourly-bin index \(h=0,\ldots,23\), synthetic hourly temperature is

\[
\widehat T_{d,h}=T_d+\max(T_d^{\max}-T_d,0)\cos\{2\pi(h+0.5)/24\}. \tag{S19}
\]

Let \(s_{d,h}=\exp\{17.625\widehat T_{d,h}/(243.04+\widehat T_{d,h})\}\). A common relative vapor pressure \(v_d\) is found by 30 bisection iterations on \([0,\max_hs_{d,h}]\), matching \(24^{-1}\sum_h\min(v_d/s_{d,h},1)=R_d/100\). Synthetic hourly humidity is \(\widehat R_{d,h}=100\min(v_d/s_{d,h},1)\), with exact zero and 100% values retained at those daily-mean boundaries. Then

\[
H_{d,h}=\mathbf1(\widehat R_{d,h}\geq90-10^{-7}),\qquad
f_d^R=1-\exp[-P_d/(24r_*)], \tag{S20}
\]

\[
e_d={1\over24}\sum_{h=0}^{23}\exp\left[-{1\over2}\left({\widehat T_{d,h}-18\over8}\right)^2\right]
\{H_{d,h}+(1-H_{d,h})f_d^R\}. \tag{S21}
\]

The selected effective precipitation intensity is \(r_*=0.45\) mm h⁻¹. It is a meteorology-only training estimate of median daily precipitation divided by observed positive-rain hours on eligible days with mean temperature above 2 °C. Synthetic humidity assumes constant vapor pressure, and the rain overlap term assumes independence of rain timing and humid bins. These are daily exposure approximations, without observed hourly humidity or leaf-wetness reconstruction. An explicit supplied exposure in \([0,1]\) overrides this operator. Missing imported pressure inherits \(M_d=0.1\); an explicit zero remains zero.

End-day latent, symptomatic and affected fractions are

\[
L_{i,d}^{\rm total}=\sum_rL_{i,r,d},\qquad
X_{i,d}=N_{i,d}+I_{i,d}+D_{i,d},\qquad
B_{i,d}^{\rm affected}=L_{i,d}^{\rm total}+X_{i,d}=1-S_{i,d}. \tag{S22}
\]

The modeled infection-event date is the first included day with \(B_{i,d}^{\rm affected}\geq\epsilon\), and the symptom-event date is the first included day with \(X_{i,d}\geq\epsilon\), where \(\epsilon=0.001\). This is a fixed detection operator, rather than a biological infection threshold. Observed positive magnitudes are reduced to zero/positive signs for fitting. A preceding zero at day \(l\) and first positive at day \(u\) imply an onset interval \((l,u]\); a missing preceding zero gives left censoring, and an all-zero history gives right censoring. Later zero returns are persistence violations and do not replace the first-positive interval.

For a finite modeled symptom day \(t\), interval-excess loss is

\[
\mathcal D(t;l,u)=\mathbf1(l\text{ finite})\max(l+1-t,0)
+\mathbf1(u\text{ finite})\max(t-u,0). \tag{S23}
\]

A missing modeled event with an observed positive receives \(\max(30,d_{\rm end}+1-u)\) d loss; a missing event in an all-zero history is compatible through the final assessment. The observation operator targets first detectable modeled symptoms. It does not identify observed infection dates or fit disease-percentage magnitude. For assessment or interval rows, the hierarchy weight is \([K\,n_f(k)\,n_\ell(k,f)\,n_r(k,f,\ell)]^{-1}\), where \(K\) is the number of coordinate-year groups, \(n_f\) their field counts, \(n_\ell\) the represented leaf counts within a field, and \(n_r\) the rows within a leaf. Disease selection uses three inner location folds of the original 28 fields, a 144-setting grid, pooled top-three two-sided interval loss, balanced assessment-error tie breaking and fixed enumeration. The selected secondary coefficient lies at the tested upper bound. Effective coefficients remain conditional on the fixed source normalization, daily step and ecological assumptions; the fit does not identify separate biological source contributions.



### Normalized affected-tissue duration and integration support

The climate output uses symptomatic tissue \(X_{i,d}=N_{i,d}+I_{i,d}+D_{i,d}\) from Equation S22. Latent tissue is excluded. For each upper leaf, the end-of-day loss fraction and nominal reference area are

\[
f_{i,d}=X_{i,d}\mathbf{1}\{\tau_i^{\rm symptom}\text{ exists and }d\geq\tau_i^{\rm symptom}\},
\qquad
\mathcal A_{i,d}=a_{i,d}/3,\quad i=0,1,2. \tag{S24}
\]

Here the three leaves represent anatomical F1–F3 in the host model, each with a maximum one-third share of the normalized unit reference canopy. The damage mapping is an accounting assumption and is not a measured functional-green-area conversion. The daily flag-leaf assessment replay compares \(100X_{0,d}\) with the recorded disease percentage; it does not apply a physiological conversion or infer a measured leaf-area denominator.

For the inclusive start-to-terminal interval \(\mathcal W_h=\{d:\tau_{65,h}\leq d\leq\tau_{85,h}\}\), the exact daily rectangle sum is

\[
L_{wh}(\mathcal W_h)=\sum_{d\in\mathcal W_h}\sum_{i=0}^{2}\mathcal A_{h,i,d}f_{wh,i,d}(1\ \mathrm{day}). \tag{S25}
\]

Host area is evaluated for the modelled day and the disease fraction after that day's update. Thus Equation S25 is the implemented discrete form of the main-text integral (Equation 1). The reference integral replaces \(f\) with one; relative normalized damage divides the affected-tissue integral by that reference integral. Units of \(L\) are equivalent days of the nominal unit reference canopy affected. Missing or incomplete required seasonal trajectories remain unavailable.

Each whole-window factorial corner uses its host trajectory's own \(\mathcal W_h\). The shared-time setting replaces this interval, in all four corners, by \(\mathcal W_0\cap\mathcal W_1\) in elapsed days since sowing. It retains the same tissue dynamics, symptom gate and nominal area. The ungated setting instead uses \(f_{i,d}=X_{i,d}\) throughout the corresponding window. Reference-only linear senescence multiplies \(\mathcal A_{i,d}\) by a factor declining linearly from one at the start gate to zero at the terminal gate, without changing simulated tissue states. The stem-extension diagnostic starts at \(\tau_{31}\). No grain-yield conversion is applied to these outputs.


### Separate rolling monitoring correction

The monitoring reference is an observation-conditioned statistical model. Its unfiltered forecast \(p_i^0\), in percent, is the equal-weight average of a fitted process/history expert and the selected station/event-history expert. The selection depends on strictly earlier field information. The two branches and their fitted backbone models retain their original source definitions and inherited crop-feature ancestry; this statistical backbone is not the frozen climate disease fit.

Let \(g(p)=\operatorname{logit}\{\min(0.995,\max(0.005,p/100))\}\), \(b_i=g(p_i^0)\), and \(r_i=g(y_i)-b_i\). Native source-code equality is categorical, without implying anatomical rank equivalence. Within one field, the covariance components are

\[
k_f(u)=a_f^2(1+\sqrt{3}u/\ell_f)\exp(-\sqrt{3}u/\ell_f). \tag{S26}
\]

\[
K_{ij}=k_f(|t_i-t_j|)+a_c^2\mathbf{1}(c_i=c_j)\exp(-|t_i-t_j|/\ell_c). \tag{S27}
\]

Here \(c_i\) is the native code, \(a_f,a_c>0\) are relative amplitudes, and \(\ell_f,\ell_c\) are time scales in days. For query date \(q\), \(C_q\) contains only strictly earlier dates from that field; all source codes observed on the query day are withheld together. The correction and forecast are

\[
\widehat r_q=K_{qC_q}(K_{C_qC_q}+I)^{-1}r_{C_q},
\qquad
\widehat y_q=100\operatorname{logit}^{-1}(b_q+\widehat r_q). \tag{S28}
\]

The unit diagonal term fixes the covariance scale. It is an effective smoothing convention, not an identified observation-noise variance or a replicate sampling denominator. If \(C_q\) is empty, the forecast is exactly \(p_q^0\), including its original endpoints. Neither current-season observations from other fields nor query-day disease enters this correction. Weather ends on the preceding day. The site-resampling intervals in Figure 5 describe fixed-forecast calibration means, not Gaussian predictive intervals from Equation S28.

With \(\theta=\log(a_f,a_c,\ell_f,\ell_c)\) and \(\theta_0=\log(1,1,30,30)\), fitting minimizes

\[
J(\theta)=\sum_i w_i\{(\widehat y_i(\theta)-y_i)/100\}^2
+10^{-5}\|\theta-\theta_0\|_2^2. \tag{S29}
\]

Weights \(w_i\) are normalized scientific hierarchy weights within the historical outer-training population. Backbone forecasts for this prequential objective exclude the held training coordinates. Amplitudes are bounded to 0.01–100 and time scales to 2–180 days; optimization uses L-BFGS-B with at most 60 iterations. The 20 retained outer-fold parameter vectors and source identity are in parameters/monitoring_discrepancy_parameters.json. These historical adaptive comparisons do not establish independent end-to-end accuracy of every inherited feature or the autonomous climate model.
