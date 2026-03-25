# Turn FOG References

This project's turn-focused FOG heuristics are intended as screening features, not diagnostic criteria.

## Design Rationale

The current turn analysis emphasizes five signals:

1. Turn-triggered context
2. Reduced effective stepping during turning
3. Poor step alternation during turning
4. Short shuffling steps during turning
5. Mismatch between body rotation and foot progression
6. Brief stalls while rotation is still ongoing

These choices are based on published descriptions of freezing of gait in Parkinson's disease:

- Turning is one of the most common contexts in which FOG appears.
- FOG is defined as a brief episodic inability to generate effective stepping.
- Clinical descriptions of FOG commonly include short shuffling steps and leg trembling.
- Patients with PD and FOG show stronger impairment when gait must be initiated or modulated.

## References

1. Nutt JG, Bloem BR, Giladi N, Hallett M, Horak FB, Nieuwboer A. Freezing of gait: moving forward on a mysterious clinical phenomenon. Lancet Neurol. 2011;10(8):734-744. DOI: 10.1016/S1474-4422(11)70143-0. PubMed: https://pubmed.ncbi.nlm.nih.gov/21777828/

2. Palmisano C, et al. Gait Initiation Impairment in Patients with Parkinson's Disease and Freezing of Gait. Bioengineering (Basel). 2022;9(11):639. DOI: 10.3390/bioengineering9110639. PubMed: https://pubmed.ncbi.nlm.nih.gov/36354550/

3. Okada Y, Fukumoto T, Takatori K, Nagino K, Hiraoka K. Abnormalities of the First Three Steps of Gait Initiation in Patients with Parkinson's Disease with Freezing of Gait. Parkinsons Dis. 2011;2011:202937. DOI: 10.4061/2011/202937. PubMed: https://pubmed.ncbi.nlm.nih.gov/22135799/

4. Spildooren J, et al. Turning and unilateral cueing in Parkinson's disease patients with and without freezing of gait. Neuroscience. 2012;207:298-306. DOI: 10.1016/j.neuroscience.2012.01.024. PubMed: https://pubmed.ncbi.nlm.nih.gov/22342993/

## Mapping From Literature to Code

- `low_motion_ratio`:
  proxy for ineffective stepping during a turn.
  Evidence level: direct concept support from the clinical definition of FOG as impaired effective stepping.

- `shuffle_step_score`:
  proxy for short shuffling stepping often seen in FOG episodes.
  Evidence level: direct clinical-description support.

- `step_irregularity_score`:
  proxy for disrupted left-right stepping alternation and repeated failed stepping attempts.
  Evidence level: indirect heuristic derived from reports of trembling legs, repeated APAs, and impaired coupling from posture to stepping.

- `rotation_motion_mismatch_score`:
  proxy for cases where the body is rotating but foot progression does not follow effectively.
  Evidence level: indirect heuristic derived from turn-triggered FOG and ineffective stepping during gait modulation.

- `stall_during_rotation_score`:
  proxy for brief stepping stalls that occur while turn-related angular motion is still present.
  Evidence level: indirect heuristic derived from the concept of ineffective stepping during an ongoing gait-modulation demand.

- `turn_context`:
  ensures turn scoring is interpreted separately from standing or straight walking segments.
  Evidence level: direct trigger-context support.

## What Is Directly Supported vs Heuristic

Directly supported by the cited literature:

- turning is a common trigger of FOG
- FOG involves failure to generate effective stepping
- short shuffling steps and trembling are clinically relevant descriptors
- gait initiation and gait modulation are especially impaired in freezers

Heuristic engineering choices in this codebase:

- exact score weights
- threshold values used to call `possible_fog_turn`
- the formula for `rotation_motion_mismatch_score`
- the proxy used for left-right alternation failure
- the proxy used for shuffle-like stepping from monocular pose velocity

These heuristic components were chosen to stay consistent with the literature, but they are not themselves taken verbatim from a validated clinical scoring system.

## Important Limitation

These features are literature-informed heuristics built on monocular pose estimation. They are not validated clinical biomarkers on their own. Any medical interpretation requires validation on labeled turning-FOG datasets.
