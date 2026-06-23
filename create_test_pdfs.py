from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

def create_pdf(filename, title, content_paragraphs):
    # Ensure directory exists
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph(title, styles['Heading1']))
    story.append(Spacer(1, 12))
    
    for p_text in content_paragraphs:
        story.append(Paragraph(p_text, styles['Normal']))
        story.append(Spacer(1, 8))
        
    doc.build(story)
    print(f"Created PDF: {filename}")

if __name__ == "__main__":
    # Create standards directory if not exists
    os.makedirs("data/standards", exist_ok=True)
    
    # 1. Create standard tia_942b.pdf
    create_pdf(
        "data/standards/tia_942b.pdf",
        "TIA-942-B Telecommunications Infrastructure Standard for Data Centers",
        [
            "Clause 5.3 Electrical Systems.",
            "Clause 5.3.4 UPS Redundancy Requirement.",
            "For Tier III and Tier IV data center configurations, uninterruptible power supply (UPS) systems must have at least N+1 redundancy to prevent outages during maintenance or component failure.",
            "Clause 5.3.5 Battery Autonomy Requirement.",
            "The minimum autonomy battery backup time for active data center loads is 15 minutes at full rated load under standard operating conditions.",
            "Clause 5.4 Mechanical Systems and Temperature Range.",
            "Clause 5.4.1 Operating Temperature.",
            "Operating temperature in the computer room must be maintained between 18 degrees Celsius and 27 degrees Celsius. Relative humidity must be between 30% and 60%."
        ]
    )

    # 2. Create submittal ups_submittal_fail.pdf
    create_pdf(
        "ups_submittal_fail.pdf",
        "Vendor Submittal: Uninterruptible Power Supply (UPS) System",
        [
            "Project: Mumbai DC Submittal.",
            "Equipment: Eaton Power-DC 120kVA UPS.",
            "Technical Specifications:",
            "- Model: Eaton Power-DC 120kVA",
            "- Capacity: 120kVA total active power load",
            "- Battery Type: Sealed Lead-Acid",
            "- Battery Backup Run Time: 10 minutes autonomy backup time at full load.",
            "- System Configuration and Redundancy: Single Module Configuration, N redundancy (No parallel redundancy or redundant module provided).",
            "- Output Voltage: 400V AC +/- 1%",
            "- Operating Temperature: 0 degrees Celsius to 40 degrees Celsius."
        ]
    )
