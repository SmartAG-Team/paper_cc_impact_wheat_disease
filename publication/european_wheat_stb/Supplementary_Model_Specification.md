## Supplementary Methods: mathematical specification

Climate calculations link wheat development to Septoria tritici blotch through leaf appearance, unfolding and seasonal weather exposure. The disease model predicts infection, progression and leaf damage on the supplied crop trajectories. Disease damage does not feed back into crop-development rates. The canopy and yield calculations map that damage to assumed functional leaf-area loss and then to a disease-related yield index; they do not constitute a complete grain-yield model.

### Calendar, state boundary and developmental clocks

The daily forcing comprises mean temperature \(T_d\) and maximum temperature \(T_d^{\max}\) in °C, mean relative humidity \(R_d\) in percent, and precipitation \(P_d\) in mm. The crop indicator \(c_d\) equals one from the supplied sowing date through the supplied crop-end date, inclusive. Regional seasons end at the first predicted BBCH85 crossing. The state indexed by \(d\) is the state at the end of day \(d\); initial developmental clocks are zero. Weather outside the crop window advances the calendar while freezing developmental clocks, tissue and local source potential. The forcing is consecutive, and a fresh simulation includes the sowing date. Accepted daily mean temperatures are at most 40 °C, the upper support of the retained thermal response.

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

### Canopy damage, healthy-area duration and the yield estimate

For F1–F3, an explicitly supplied nominal maximum upper-three-leaf reference LAI \(L_*\) defines the scenario area

\[
\mathcal A_{i,d}=L_*a_{i,d}/3,\quad i=0,1,2. \tag{S24}
\]

\(\mathcal A\) has units m² lamina m⁻² ground; \(L_*=1\) in the standardized scenario. The public daily runtime and field full-season replay use \(f_{i,d}=X_{i,d}\) as the assumed functional-loss fraction. The published climate replay applies a detection gate, \(f_{i,d}=X_{i,d}\mathbf1(\tau_i^{\rm symptom}\text{ exists and }d\geq\tau_i^{\rm symptom})\). Climate subthreshold damage therefore contributes zero before the first detected symptom, whereas the field replay and public runtime retain that subthreshold contribution. Latent tissue is excluded in both definitions. Neither mapping is a measured functional-green-area conversion.

With the inclusive flowering-proxy–BBCH85 window \(\mathcal W=\{d:\tau_{65}\leq d\leq\tau_{85}\}\), daily rectangular integration gives

\[
\mathrm{HAD}_{3}^{\rm ref}=\sum_{d\in\mathcal W}\sum_{i=0}^{2}\mathcal A_{i,d}(1\ \mathrm d),\qquad
\mathrm{HAD}_{3}^{\rm lost}=\sum_{d\in\mathcal W}\sum_{i=0}^{2}\mathcal A_{i,d}f_{i,d}(1\ \mathrm d), \tag{S25}
\]

\[
\mathrm{HAD}_{3}^{\rm healthy}=\mathrm{HAD}_{3}^{\rm ref}-\mathrm{HAD}_{3}^{\rm lost},\qquad
\Delta Y_b=b\,\mathrm{HAD}_{3}^{\rm lost}. \tag{S26}
\]

HAD units are m² green lamina m⁻² ground d, equivalent to GLAI d. Normalized HAD loss is \(\mathrm{HAD}_{3}^{\rm lost}/L_*\), with units of days. The climate figures and tables report this normalized quantity. Its numerical value equals unnormalized HAD loss in the unit-LAI scenario. Relative HAD loss is \(\mathrm{HAD}_{3}^{\rm lost}/\mathrm{HAD}_{3}^{\rm ref}\), expressed as a percentage. The slope scenarios \(b=0.0141,0.0180,0.0207\) have units t ha⁻¹ per GLAI d. Their products have units t ha⁻¹, conditional on the nominal reference canopy and conversion; the unit-LAI scenario scales linearly with \(L_*\). Equation S26 uses \(\Delta Y_b\) for a positive disease-related loss index. The future-minus-historical yield change has the opposite sign: \(-b(\mathrm{HAD}_{3,\rm future}^{\rm lost}-\mathrm{HAD}_{3,\rm historical}^{\rm lost})\). Reported yield changes per unit reference LAI divide this expression by \(L_*\). Thus an increase in HAD loss gives a negative estimated yield change. The source coefficient window began at GS31, with a different integration convention and unrecovered exact terminal rule; the flowering–85 diagnostic is a restricted-window adaptation. A separate climate GS31–85 diagnostic uses the same definitions with start date \(\tau_{31}\). Slope ranges are cultivar scenarios, rather than confidence intervals. Incomplete windows yield missing conditional losses. Percentage loss is \(100\Delta Y_b/Y_{\rm ref}\) only when an independent positive reference yield \(Y_{\rm ref}\) is supplied. No yield is inferred from nominal LAI, and losses above an independent reference yield are flagged without clipping. An external-area runtime mode integrates caller-supplied reference LAI and functional-loss fractions instead. Actual field-yield forecasts remain undefined.

### Full-grid aggregation and sampled diagnostics

The main climate experiment simulates all 14,941 wheat land-use cells, with eligible fixed winter-wheat rainfed calendars for 14,932 cells. Three climate models, three emissions pathways and three 30-year periods give 810 annual jobs. Full-grid regional and country estimates use fixed harvested-area weights across valid historical–future season pairs. Equations S27 and S28 apply to these estimates with the draw index replaced by a grid-cell index and weights equal to each cell’s harvested area divided by the total harvested area in the reporting domain. Complete spatial coverage eliminates spatial-sampling Monte Carlo error, while unavailable seasons and other uncertainties remain explicit.

The separate weather–wheat-development decomposition and sensitivity analysis use 16 geographical strata, four independent draws with replacement per stratum, and selection probability proportional to registered harvested area within each stratum. The 64 draw identities comprise 62 unique cells; coincident cells retain separate draw identities. If \(W_h\) is eligible area in stratum \(h\) divided by total eligible area, each of its \(n_h=4\) draws has weight \(w_{hr}=W_h/n_h\). Missing calendars are excluded from the eligible population; missing weather, unreached BBCH85 and unsupported temperatures yield missing outcomes. Calendar type and actual wheat genetics remain scenario assumptions.

For climate model \(g\), SSP \(s\), period \(p\), draw \(r\), and relative year \(t=1,\ldots,30\), let \(y_{gsprt}\) denote a metric and \(v_{gsprt}\) indicate a finite metric. Define

\[
x_{gspr}={1\over30}\sum_t v_{gsprt}y_{gsprt},\qquad
z_{gspr}={1\over30}\sum_t v_{gsprt},\qquad
\widehat c_{gsp}=\sum_rw_rz_{gspr},\qquad
\widehat\mu_{gsp}={\sum_rw_rx_{gspr}\over\widehat c_{gsp}}. \tag{S27}
\]

The numerator convention assigns zero contribution to missing values, including when all 30 values for a draw are missing; \(\widehat c\) is metric-specific area-time coverage. Period estimates are ratios of equal-year area-time totals. They are not averages of separately normalized annual means. Event-date metrics condition on detected events because absent events remain missing.

Historical and future outcomes are paired within the same draw, climate model and SSP at the same relative-year position. Let \(a_{rt}\) and \(b_{rt}\) be historical and future metrics, and \(v_{rt}^{\cap}=\mathbf1(a_{rt}\text{ and }b_{rt}\text{ finite})\). Then

\[
x_r^{\Delta}={1\over30}\sum_t v_{rt}^{\cap}(b_{rt}-a_{rt}),\qquad
z_r^{\cap}={1\over30}\sum_t v_{rt}^{\cap},\qquad
\widehat\Delta={\sum_rw_rx_r^{\Delta}\over\sum_rw_rz_r^{\cap}}. \tag{S28}
\]

Historical and future means on this same common support satisfy \(\widehat\Delta=\widehat\mu_b^{\cap}-\widehat\mu_a^{\cap}\). Pairing aligns positions in 30-year periods, without asserting a physical correspondence between individual historical and future years.

For any ratio estimate \(\widehat\mu=\sum_rw_rx_r/\widehat c\), let \(\psi_r=(x_r-\widehat\mu z_r)/\widehat c\). For the sampled diagnostic analysis, the spatial Monte Carlo standard error is

\[
\widehat{\operatorname{MCSE}}(\widehat\mu)=
\left[\sum_h{W_h^2\over n_h}s_h^2(\psi)\right]^{1/2},\qquad
s_h^2(\psi)={1\over n_h-1}\sum_{r\in h}(\psi_r-\bar\psi_h)^2. \tag{S29}
\]

The same expression uses \(x_r^{\Delta},z_r^{\cap}\) for paired changes. No finite-population correction applies to independent replacement draws. The three-model ensemble mean is the mean of model-specific ratios. Because every climate model shares spatial draws, its MCSE uses \(\bar\psi_r=3^{-1}\sum_g\psi_{gr}\) inside Equation S29, retaining cross-model spatial covariance. Model minima and maxima describe three-model spread, rather than confidence intervals. Spatial MCSE quantifies simulation-sampling error conditional on the registered forcing, fit and scenarios; it does not include parameter, observation, source-process, calendar or climate-projection uncertainty. The archived severity-model parameter ensembles and full-grid anchors have a different model identity and do not supply uncertainty intervals for the current source/tissue model.

### Model identity and evidence scope

The continental climate calculations use `simulate_draw_seasons` in `analysis/paper_study/overwinter_leaf_model_20261006/climate/run.py` through the full-grid driver. The public daily interface is `model.wheat_stb` version 0.1.0. Both use the source/tissue kernel in `model/seasonal_septoria/overwinter.py`, stage-anchored canopy in `leaf_phenology.py`, and daily exposure operator in `wetness.py`. The current selected disease fit is `overwinter_source_model/frozen_selected_fit.json`; the ordered stage fit and retained donor TPV record are separate explicit inputs. The earlier `seasonal_calibration_v1` severity formulation and `infection_and_upper3_yield_v1_20261006` event formulation retain their original identities and archived outcomes. Only the stage-calibrated flowering threshold is inherited from the latter. Source hashes, configuration hashes and the parameter inventory identify exact implementations and input records. Numerical consistency of the equations establishes computational correspondence, without establishing predictive accuracy, measured source identification, observed infection dates or actual yield loss.
