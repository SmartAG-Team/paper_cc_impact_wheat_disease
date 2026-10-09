# Public-source attribution and license

Original data author: **Björn Andersson**. Both versioned Figshare datasets are licensed **Creative Commons Attribution 4.0 International (CC BY 4.0)**, as recorded in the included original repository metadata. License terms: <https://creativecommons.org/licenses/by/4.0/>.

- Andersson, Björn (2022). *Data set for weather, development stage and yield*. Figshare. [10.6084/m9.figshare.19203377.v4](https://doi.org/10.6084/m9.figshare.19203377.v4).
- Andersson, Björn (2022). *SAS script and input files*. Figshare. [10.6084/m9.figshare.19203398.v3](https://doi.org/10.6084/m9.figshare.19203398.v3).

`nordic_public_sources.zip` contains nine exact, unmodified archive members: the yield, development-stage and weather-station workbooks; both SAS workbooks; the SAS example script; the weather-description document; and both repository metadata records. Public downloadable file MD5 values agree with the repository metadata. SHA-256 values, file sizes, file identifiers, download URLs and archive-member paths are in `manifest.json`. Each selected member was byte-identical in the supplied consolidated and research-source archives. Those archives were read only.

`Weather data.txt` is outside this subset because no raw-weather analysis is executed. Its repository metadata and download URL remain in the unmodified metadata record. The ZIP subset is 167,763 bytes; standalone numerical reproduction requires this subset, not either original archive.

The tables in `../outputs/` are adaptations: standardized fields, auditable exact-key attachments, source-row exclusions, paired grain differences and statistical summaries. These changes are implemented in `../run_analysis.py`. Adapted data retain attribution to the original datasets under CC BY 4.0; this attribution makes no claim of endorsement by the source author. The dataset license does not by itself establish a license for unrelated repository code.
