#!/usr/bin/env python3
"""Download all literature papers referenced in the project."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

import httpx
import time

PAPERS_DIR = project_root / "data" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# All papers referenced in the project
PAPERS = [
    # 1. CARE-PD main paper
    {
        "id": "arxiv_2311.09890",
        "title": "CARE-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment",
        "authors": "Adeli V et al.",
        "year": 2025,
        "venue": "NeurIPS 2025",
        "doi": "10.48550/arXiv.2311.09890",
        "arxiv_id": "2311.09890",
        "download_url": "https://arxiv.org/pdf/2311.09890.pdf",
    },
    # 2. 3DGait - Video-Based Gait Analysis for Alzheimer's/DLB
    {
        "id": "doi_10.1007_978-3-031-47076-9_8",
        "title": "Video-Based Gait Analysis for Assessing Alzheimer's Disease and Dementia with Lewy Bodies",
        "authors": "Wang D, Zouaoui C, Jang J, Drira H, Seo H",
        "year": 2023,
        "venue": "AMAI 2023 (MICCAI Workshop)",
        "doi": "10.1007/978-3-031-47076-9_8",
        "download_url": None,  # Springer, try Unpaywall
    },
    # 3. BMCLab - PD Walking Kinematics/Kinetics
    {
        "id": "pmid_36875659",
        "title": "A public data set of walking full-body kinematics and kinetics in individuals with Parkinson's disease",
        "authors": "Shida TKF, Costa TM, de Oliveira CEN et al.",
        "year": 2023,
        "venue": "Front Neurosci",
        "doi": "10.3389/fnins.2023.992585",
        "pmid": "36875659",
        "pmcid": "PMC9978741",
        "download_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9978741/pdf/",
    },
    # 4. DNE Paper 1 - Smartphone-Based Digitized Neurological Examination
    {
        "id": "pmid_39186431",
        "title": "Smartphone-Based Digitized Neurological Examination Toolbox for Multi-test Neurological Abnormality Detection",
        "authors": "Hoang TH, Zallek C, Do MN",
        "year": 2024,
        "venue": "IEEE J Biomed Health Inform",
        "doi": "10.1109/JBHI.2024.3439492",
        "pmid": "39186431",
        "download_url": None,
    },
    # 5. DNE Paper 2 - Vision-Based Digitized Neurological Examination
    {
        "id": "pmid_35439148",
        "title": "Towards a Comprehensive Solution for a Vision-Based Digitized Neurological Examination",
        "authors": "Hoang TH, Zehni M, Xu H, Heintz G, Zallek C, Do MN",
        "year": 2022,
        "venue": "IEEE J Biomed Health Inform",
        "doi": "10.1109/JBHI.2022.3167927",
        "pmid": "35439148",
        "pmcid": "PMC9707344",
        "download_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9707344/pdf/",
    },
    # 6. E-LC Paper 1 - Freezing of Gait persistence after levodopa
    {
        "id": "pmid_31799377",
        "title": "Freezing of Gait can persist after an acute levodopa challenge in Parkinson's disease",
        "authors": "Lucas McKay J, Goldstein FC, Sommerfeld B et al.",
        "year": 2019,
        "venue": "NPJ Parkinsons Dis",
        "doi": "10.1038/s41531-019-0099-z",
        "pmid": "31799377",
        "pmcid": "PMC6874572",
        "download_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6874572/pdf/",
    },
    # 7. E-LC Paper 2 - Spatial-Temporal GCN for Freezing of Gait
    {
        "id": "pmid_36850363",
        "title": "An Explainable Spatial-Temporal Graphical Convolutional Network to Score Freezing of Gait in Parkinsonian Patients",
        "authors": "Kwon H, Clifford GD, Genias I et al.",
        "year": 2023,
        "venue": "Sensors",
        "doi": "10.3390/s23041766",
        "pmid": "36850363",
        "pmcid": "PMC9968199",
        "download_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9968199/pdf/",
    },
    # 8. KUL-DT-T Paper 1 - Freezing of Gait dual-tasking/turning
    {
        "id": "pmid_20632376",
        "title": "Freezing of gait in Parkinson's disease: the impact of dual-tasking and turning",
        "authors": "Spildooren J, Vercruysse S, Desloovere K et al.",
        "year": 2010,
        "venue": "Mov Disord",
        "doi": "10.1002/mds.23327",
        "pmid": "20632376",
        "download_url": None,  # Wiley, try Unpaywall
    },
    # 9. KUL-DT-T Paper 2 - Automated FoG assessment with GCN
    {
        "id": "pmid_35597950",
        "title": "Automated freezing of gait assessment with marker-based motion capture and multi-stage spatial-temporal graph convolutional neural networks",
        "authors": "Filtjens B, Ginis P, Nieuwboer A, Slaets P, Vanrumste B",
        "year": 2022,
        "venue": "J Neuroeng Rehabil",
        "doi": "10.1186/s12984-022-01025-3",
        "pmid": "35597950",
        "pmcid": "PMC9124420",
        "download_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9124420/pdf/",
    },
    # 10. PD-GaM Paper 1 - GAITGen
    {
        "id": "arxiv_2503.22397",
        "title": "GAITGen: Disentangled motion-pathology impaired gait generative model",
        "authors": "Adeli V, Mehraban S, Mirmehdi M et al.",
        "year": 2025,
        "venue": "arXiv preprint",
        "arxiv_id": "2503.22397",
        "download_url": "https://arxiv.org/pdf/2503.22397.pdf",
    },
    # 11. PD-GaM Paper 2 - PeCop
    {
        "id": "pecop_wacv2024",
        "title": "PeCop: Parameter efficient continual pretraining for action quality assessment",
        "authors": "Dadashzadeh A, Duan S, Whone A, Mirmehdi M",
        "year": 2024,
        "venue": "IEEE/CVF WACV 2024",
        "doi": "10.1109/WACV57701.2024.00011",
        "download_url": None,
    },
    # 12. MDS-UPDRS reference
    {
        "id": "goetz_2008_mds_updrs",
        "title": "Movement Disorder Society-sponsored revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS)",
        "authors": "Goetz CG et al.",
        "year": 2008,
        "venue": "Mov Disord",
        "doi": "10.1002/mds.22340",
        "pmid": "19025984",
        "download_url": None,
    },
    # 13. Hoehn and Yahr Scale
    {
        "id": "hoehn_yahr_1967",
        "title": "Parkinsonism: onset, progression and mortality",
        "authors": "Hoehn MM, Yahr MD",
        "year": 1967,
        "venue": "Neurology",
        "doi": "10.1212/WNL.17.5.427",
        "pmid": "6067254",
        "download_url": None,
    },
    # 14. House-Brackmann Scale
    {
        "id": "house_brackmann_1985",
        "title": "Facial nerve grading system",
        "authors": "House JW, Brackmann DE",
        "year": 1985,
        "venue": "Otolaryngol Head Neck Surg",
        "doi": "10.1177/019459988509300202",
        "pmid": "2984979",
        "download_url": None,
    },
]


def download_pdf(url: str, output_path: Path, client: httpx.Client) -> bool:
    """Download a PDF from a URL."""
    try:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        # Verify it looks like a PDF
        if b"%PDF" in response.content[:1024] or "application/pdf" in content_type:
            output_path.write_bytes(response.content)
            size_kb = len(response.content) / 1024
            print(f"    -> Downloaded: {output_path.name} ({size_kb:.0f} KB)")
            return True
        else:
            print(f"    -> Not a PDF (content-type: {content_type})")
            return False
    except Exception as e:
        print(f"    -> Download failed: {e}")
        return False


def try_unpaywall(doi: str, client: httpx.Client) -> str | None:
    """Try to get open access PDF URL from Unpaywall."""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}"
        response = client.get(url, params={"email": "research@example.com"})
        if response.status_code != 200:
            return None
        data = response.json()
        best_oa = data.get("best_oa_location")
        if best_oa and best_oa.get("url_for_pdf"):
            return best_oa["url_for_pdf"]
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except Exception:
        pass
    return None


def try_pmc_download(pmcid: str, client: httpx.Client) -> str | None:
    """Get PMC PDF URL."""
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"


def main():
    print("=" * 80)
    print("Literature Download Script")
    print(f"Target directory: {PAPERS_DIR}")
    print(f"Total papers to process: {len(PAPERS)}")
    print("=" * 80)

    client = httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (research-automation; academic use)"},
    )

    downloaded = 0
    already_exists = 0
    failed = []

    for i, paper in enumerate(PAPERS, 1):
        paper_id = paper["id"]
        output_path = PAPERS_DIR / f"{paper_id}.pdf"

        print(f"\n[{i}/{len(PAPERS)}] {paper['title']}")
        print(f"  Authors: {paper['authors']} ({paper['year']})")
        print(f"  Venue: {paper['venue']}")
        if paper.get("doi"):
            print(f"  DOI: {paper['doi']}")

        # Check if already downloaded
        if output_path.exists() and output_path.stat().st_size > 1000:
            size_kb = output_path.stat().st_size / 1024
            print(f"  -> Already exists ({size_kb:.0f} KB), skipping.")
            already_exists += 1
            continue

        success = False

        # Strategy 1: Direct download URL
        if paper.get("download_url"):
            print(f"  Trying direct URL...")
            success = download_pdf(paper["download_url"], output_path, client)

        # Strategy 2: arXiv
        if not success and paper.get("arxiv_id"):
            arxiv_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
            print(f"  Trying arXiv ({paper['arxiv_id']})...")
            success = download_pdf(arxiv_url, output_path, client)

        # Strategy 3: PMC
        if not success and paper.get("pmcid"):
            pmc_url = try_pmc_download(paper["pmcid"], client)
            if pmc_url:
                print(f"  Trying PMC ({paper['pmcid']})...")
                success = download_pdf(pmc_url, output_path, client)

        # Strategy 4: Unpaywall (open access)
        if not success and paper.get("doi"):
            print(f"  Trying Unpaywall...")
            oa_url = try_unpaywall(paper["doi"], client)
            if oa_url:
                print(f"  Found OA URL: {oa_url}")
                success = download_pdf(oa_url, output_path, client)
            else:
                print(f"    -> No open access version found")

        if not success:
            failed.append(paper)
            print(f"  ** FAILED - could not download")

        else:
            downloaded += 1

        # Be polite to servers
        time.sleep(1)

    client.close()

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"  Total papers:     {len(PAPERS)}")
    print(f"  Already existed:  {already_exists}")
    print(f"  Newly downloaded: {downloaded}")
    print(f"  Failed:           {len(failed)}")

    if failed:
        print(f"\nFailed papers (may require institutional access):")
        for p in failed:
            print(f"  - [{p['year']}] {p['title']}")
            if p.get("doi"):
                print(f"    DOI: https://doi.org/{p['doi']}")
            if p.get("pmid"):
                print(f"    PubMed: https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/")

    # List all papers in directory
    print(f"\nAll papers in {PAPERS_DIR}:")
    for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
        size_kb = pdf.stat().st_size / 1024
        print(f"  {pdf.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
