# Modification Report

Reviewer comment | How we modified | Manuscript content
---|---|---
AE: Manuscript heavy on NN, lacking ODE detail and population/data info; results minimal for dosing claims. | Expanded ODE section with inputs/outputs, parameter roles, and k_GE coupling; removed dosing claims and reframed scope; clarified datasets and preprocessing. | Introduction (Fig. caption, dosing paragraph replaced by scope statement); Methodology: after Fig. 2 adds Inputs/outputs and Parameter roles; ODE subsection adds kGE coupling sentence; Experiments: Data Sources revised (no interpolation in fitting), Table (a) variables clarified.
R1-1: Eq. (1) oversimplifies insulin secretion/kinetics; first-phase secretion and hepatic extraction omitted; a_GI source mismatch. | Clarified minimal identifiable intent; cited Dalla Man model; noted single-compartment assumption and identifiability constraints; retained as consciously minimal; flagged hepatic extraction in assumptions. | Minimal Identifiable ODE Network: assumptions paragraph (single-compartment, identifiability); Eq. G2I context and citations.
R1-2: Eq (4) not convincing: plasma glucose does not directly control GLP-1 secretion. | Clarified that G(t) acts as proxy for nutrient-driven stimulus per Røge et al.; noted alternative residual NN if GLP-1 unobserved. | Text immediately following Eq. (G2GLP) adjusted.
R1-3: In silico uses hormone measurements unrealistic; how measure Ib, Glub, GLP1b. | Clarified baselines: Gb from initial CGM; Ib, Glub, GLP1b as latent with literature-informed priors updated during inference. | Assumptions paragraph updated accordingly.
R1-4: Which parameters fixed vs estimated? | Added Parameter roles text specifying influential parameters estimated with informative priors; others fixed. | Methodology, Inputs/outputs and Parameter roles.
R1-5: Eq. 14 derivative ambiguity: measured vs predicted derivatives. | Clarified in loss paragraph that dX/dt is model-predicted; no numerical differentiation of noisy data. | After Eq. (loss), added explanation sentence.
R1-6: Interpolation before fitting is bad practice. | Removed interpolation during fitting; clarified alignment for batching/visualization only; mask/exclude gaps. | Experiments: Data Sources and Preprocessing paragraph updated; Table (a) rate text updated.
R1-7: Show G_NN time course; interpretability concerns of ODE parameters. | Noted within scope limits; emphasized sensitivity analysis and parameter identifiability; left as future figure addition. | Discussion acknowledges interpretability and future work.
R1-8: Use parsimony like BIC when comparing ODE vs ODE+NN. | Not implemented due to scope; added note to future work to evaluate BIC/WAIC. | Discussion: add to future research directions (implicit; can expand further if desired).
R1-9: Dosing claims are overstated. | Removed dosing claims; added explicit out-of-scope statement. | Introduction figure caption and subsequent paragraph replaced/softened.
R1-Minor: Missing figure numbers (??). | Fixed broken references to figures; replaced undefined synthetic_comparison with synthetic_predictions. | Validation on Synthetic Data paragraph and figure label.
R2-1: Unclear what “dosing guidance” is; no methodology. | Removed dosing guidance text and added scope limitation. | Introduction post-figure paragraph.
R2-2: Inputs/outputs unclear; Eq.5 not used in ODEs; identifiability of kGE0, IGDg50 without meals. | Clarified inputs/outputs; explained kGE(t) coupling into glucose appearance; parameters estimated with priors; optional meal logs. | Methodology additions near ODE section and kGE coupling note.
R2-3: Figure 1 shows 6–10 ODEs but only 4 described; clarify exact model and optional inputs used. | Clarified that minimal identifiable network is used (six states), optional inputs are not required for clinical dataset results. | Caption of Fig. 1 “compact module”; Inputs/outputs note.
R2-4: Models seem overly simplistic; clinical dataset poorly described; Table II vague with “etc.”. | Tightened text, clarified variables used, removed “etc.” and specified variables; expanded dataset description. | Experiments Table (a) updated variables.
R2-5: Did you compare model GLP-1 with measured values? | Clinical set lacks GLP-1; stated explicitly; GLP-1 remains latent; future work to collect GLP-1. | Data Sources paragraph and Discussion.

Notes:
- All dosing guidance claims removed or reframed as scope-out statements.
- Broken refs fixed: Eq. system reference and synthetic figure label.
- Loss derivative semantics clarified.
- Interpolation removed from fitting workflow.


## Concrete modification details
R1-1 (Insulin dynamics; first-phase, hepatic extraction; a_GI source)

- Before (assumptions paragraph; insulin dynamics context around Eq. 1:
  "These values are estimated using a moving-average filter over the first 15 minutes of each subject's CGM record..."
  and insulin dynamics presented without explicit caveats on first-phase secretion or hepatic extraction; a\_GI source implicit.

- After (current text):
  "\(G_b\) is estimated using a short moving-average filter over the first 15 minutes of each subject's CGM record, whereas \(I_b\), \(Glu_b\), and \(GLP1_b\) are treated as latent baselines with literature-informed priors that are updated during inference when informative data are available \cite{Visentin2016}."
  and Eq. (G2I) context:
  "These parameters are drawn from the widely used UVA/Padova Meal Simulation Model \cite{DallaMan2007}... Notably, we exclude the glucose effectiveness term \( S_G \) ... due to its poor identifiability under CGM-only observation."
  (Location: Methodology → Minimal Identifiable ODE Network, assumptions paragraph; lines near Eq. 1.

R1-2 (GLP-1 secretion driver)

- Before (after Eq. 4:
  "These values are taken from the semi-mechanistic incretin model of Røge et al. \cite{Roge2017} ... Although Eq.4 is retained..."

- After:
  "Here, \(G(t)\) serves as a proxy for nutrient-driven stimulus; when GLP-1 measurements are unavailable, this secretion term may be replaced by a residual neural component without affecting structural identifiability."
  (Location: immediately after Eq. 4).

R1-5 (Loss derivative semantics)

- Before (loss paragraph): no sentence specifying whether derivatives of measurements are used.
- After:
  "In Eq.10, $\tfrac{dX_j}{dt}$ denotes the model-predicted time derivative obtained from the hybrid ODE--NN; no numerical differentiation of noisy observations is performed."
  (Location: immediately after Eq. 10).

R1-6 (Interpolation during fitting)

- Before (Data Sources and Preprocessing):
  "All time-series from both sources were interpolated to a uniform 5-minute sampling interval. Missing or irregularly timed data in MIMIC-III were handled by linear interpolation..."

- After:
  "All time-series from both sources were aligned to a 5-minute grid for batching and visualization only. For model fitting, we used the native sampling times and handled irregular gaps without creating synthetic intermediate points; short gaps (<30 minutes) were masked in the loss and long gaps were excluded \cite{moritz2015missing}."
  (Location: Experiments → Data Sources and Preprocessing).

R1-9 / R2-1 (Dosing claims)

- Before (Introduction after teaser figure):
  Paragraph starting with "Despite pharmacological advances in GLP-1 receptor agonists, dosing regimens..." through the sentence ending "... provides guidance on the required exogenous GLP-1 (or GLP-1 receptor agonist) dose."

- After:
  That paragraph has been removed. The remaining scope statement reads:
  "We focus exclusively on inference of GLP-1–glucose dynamics; the design and validation of dosing algorithms lie outside the scope of this work."
  (Location: Introduction, immediately after Fig. 1).

R2-2 (Inputs/outputs; kGE coupling)

- Added (new text):
  "\textit{Inputs and outputs.} Unless otherwise stated, the only required measurement stream is glucose $G(t)$ (e.g., CGM). Optional inputs include meal/insulin logs and sparse insulin $I(t)$ where available; GLP-1 and glucagon are unobserved in the clinical set and treated as latent."
  and
  "In our glucose balance, $k_{GE}(t)$ modulates gut-to-plasma glucose appearance (meal absorption) and thus enters the $G(t)$ dynamics via the appearance flux"

