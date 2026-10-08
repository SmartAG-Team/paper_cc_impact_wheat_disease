"""Native supplementary tables of published crop-response data."""
import pandas as pd
from .published_response import HERE


def fmt(value,digits=2):
    return '—' if pd.isna(value) else f'{value:.{digits}f}'


def supplementary_tables():
    tables=[]
    frame=pd.read_csv(HERE/'parker2004_table4_yield_loss_matrix.csv')
    rows=[['Cultivar','SX95','SX96','SX97','RM95','RM96','RM97']]
    rows += [[r.cultivar,*[fmt(x) for x in r[1:]]] for r in frame.itertuples(index=False)]
    tables.append((rows,'Table S7a | Published STB yield contrasts from Parker et al. (2004), Table 4. '
        'SX denotes Starcross and RM Rosemaund. Values are protected-minus-unprotected grain yield, t ha⁻¹. '
        'All 117 reported contrasts and five negative values are retained; dashes identify 33 unreported combinations. '
        'These are treatment contrasts from three-replicate experiments rather than raw plot observations.',[100,59,59,59,59,59,59]))
    frame=pd.read_csv(HERE/'parker2004_table5_yield_slopes.csv')
    rows=[['Cultivar','Random prediction','Fixed estimate','Fixed SE']]
    rows += [[r.cultivar,fmt(r.random_effect_slope_t_ha_per_GLAI_day,4),
        fmt(r.fixed_effect_slope_t_ha_per_GLAI_day,4),fmt(r.fixed_effect_slope_SE,4)] for r in frame.itertuples()]
    tables.append((rows,'Table S7b | Published top-three-leaf yield sensitivities from Parker et al. (2004), Table 5. '
        'All coefficients have units t ha⁻¹ per GLAI-day. The population transfer centre is 0.0180; cultivar predictions '
        'span 0.0141–0.0207. Fixed estimates and their standard errors are reported for 21 cultivars. '
        'Random predictions share environments and are not independent replications.',[130,114,102,92]))
    frame=pd.read_csv(HERE/'foulkes2006_group_separated_predictions.csv')
    rows=[['Genotype','Disease exposure','Protected HAD','Affected HAD','Protected yield','Affected yield']]
    labels={'STB':'STB','stripe rust':'Yellow rust','STB;stripe rust':'STB + yellow rust'}
    for r in frame.itertuples():
        rows.append([r.genotype,labels[r.genotype_observed_target_diseases],fmt(r.healthy_HAD_GLAI_day,1),
            fmt(r.diseased_HAD_GLAI_day,1),fmt(r.healthy_yield_t_ha_15pct_moisture),fmt(r.diseased_yield_t_ha_15pct_moisture)])
    tables.append((rows,'Table S8a | Foulkes et al. (2006), Table 2: paired mixed-model genotype-by-fungicide predicted means. '
        'HAD integrates the top five leaves from anthesis to complete canopy senescence, in GLAI-days. '
        'Yield is t ha⁻¹ at 85% dry matter. All eight contrasts are retained: two STB-only, two yellow-rust-only and four mixed-exposure means. '
        'The four site-years have unbalanced genotype and disease coverage. This broader canopy–yield evidence is not STB-specific epidemic validation. '
        'Average genotype-by-fungicide SED is 12.9 GLAI-days and 0.52 t ha⁻¹; these are not row-specific standard errors.',[120,64,68,68,69,69]))
    rows=[['Background/line','HAD loss (%)','Yield loss (%)','Held prediction (%)','Yield loss (t ha⁻¹)','Held prediction (t ha⁻¹)']]
    for r in frame.itertuples():
        rows.append([r.genotype,fmt(100*r.derived_HAD_loss_fraction),fmt(100*r.derived_yield_loss_fraction),
            fmt(100*r.relative_held_out_prediction),fmt(r.derived_yield_loss_t_ha,3),fmt(r.absolute_held_out_prediction,3)])
    tables.append((rows,'Table S8b | Derived published contrasts and leave-one-genetic-background-out predictions. '
        'All eight disease-labelled contrasts are included. Each coefficient uses only the other four backgrounds. Relative losses divide protected-minus-affected differences '
        'by protected means; absolute response uses top-five HAD loss. Mercia, M. Huntsman, Hobbit, Weston and Chaucer '
        'receive equal total weights. Predictions reconstruct aggregate treatment means, rather than current-model field yield.',[117,61,61,72,76,78]))
    frame=pd.read_csv(HERE/'published_yield_response_metrics.csv')
    rows=[['Response','MAE','RMSE','Bias','R²','Full-data coefficient']]
    for r in frame.itertuples():
        scale=100 if r.response=='relative' else 1
        rows.append([r.response.capitalize(),fmt(scale*r.group_balanced_MAE,3),fmt(scale*r.group_balanced_RMSE,3),
            fmt(scale*r.group_balanced_bias,3),fmt(r.group_balanced_R2,3),fmt(r.full_data_slope,5)])
    tables.append((rows,'Table S8c | Group-balanced prediction metrics for eight published mean contrasts in five backgrounds. '
        'Relative MAE, RMSE and bias are percentage points; absolute metrics are t ha⁻¹. The full-data absolute coefficient '
        'is t ha⁻¹ per top-five GLAI-day, and the relative coefficient is dimensionless. Full-data coefficients are descriptive '
        'fits; metric calculations use held-background predictions. No covariance-adjusted confidence intervals are identified.',[78,63,65,63,63,118]))
    frame=pd.read_csv(HERE/'published_yield_domain_compatibility.csv')
    rows=[['Study','Disease','Leaf scope','Predictor / outcome','Evidence role']]
    compact={'Parker2004':['Parker 2004','STB','Top three','GLAI-days / t ha⁻¹','Top-three transfer'],
        'Foulkes2006':['Foulkes 2006','STB; yellow rust','Top five','GLAI-days / t ha⁻¹','Aggregate group holdout'],
        'Bryson1997':['Bryson 1997','Yellow rust','Canopy; fixed ranks unverified','HAD; absorbed radiation / grain dry matter','Reported within-year fits'],
        'SubbaRao1989':['Subba Rao 1989','Leaf rust','Top four individually','Relative leaf HAD / tiller grain weight','Reported leaf-rank fits']}
    rows += [compact[key] for key in frame.source if key in {'Parker2004','Foulkes2006'}]
    tables.append((rows,'Table S9 | Compatibility of the retained published crop-response evidence. Disease exposure, leaf scope, '
        'integration window, predictor scale and grain-yield definition differ. The Foulkes dataset includes yellow-rust and mixed-disease contrasts. '
        'Study-specific observations, temporal scopes, units and limitations are retained in Source Data.',[78,69,93,115,104]))
    frame=pd.read_csv(HERE/'bryson1997_reported_yield_fits.csv')
    rows=[['Year','Treatment combinations','Canopy predictor','Reported R²']]
    rows += [[str(r.year),str(r.reported_treatment_combinations),r.predictor,fmt(r.value)] for r in frame.itertuples()]
    tables.append((rows,'Table S10a | Published yellow-rust grain-yield fits from Bryson et al. (1997). '
        'Healthy-area absorption weights functional canopy area by absorbed radiation. Values are within-year fitted R² '
        'from the original experiments; raw paired values and fixed leaf-rank scope were not recovered.',[57,110,175,100]))
    frame=pd.read_csv(HERE/'subbarao1989_reported_yield_fits.csv')
    rows=[['Season','Crop status','Predictor','Adjusted R²']]
    rows += [[r.crop_season,r.crop_status,'Leaf-specific relative HAD',fmt(r.value)] for r in frame.itertuples()]
    tables.append((rows,'Table S10b | Published leaf-rust tiller-grain-weight fits from Subba Rao et al. (1989). '
        'The original two-season, single-cultivar experiment includes ten leaf-retention treatments and the flag through '
        'fourth leaves. Values are fitted adjusted R². Relative duration and tiller grain weight cannot supply a '
        'stand-scale coefficient in t ha⁻¹ per GLAI-day.',[68,100,174,100]))
    return tables
