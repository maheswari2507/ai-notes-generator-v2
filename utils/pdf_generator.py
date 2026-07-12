from flask import make_response
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from io import BytesIO


def generate_pdf(overall_summary, paragraphs, key_points):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    title_style.textColor = HexColor("#2563EB")

    heading_style = styles["Heading1"]
    heading_style.textColor = HexColor("#1E40AF")

    body_style = styles["BodyText"]
    body_style.leading = 22
    body_style.spaceAfter = 12

    bullet_style = styles["BodyText"]
    bullet_style.leading = 18

    story = []

    # --------------------------------
    # Title
    # --------------------------------

    story.append(
        Paragraph(
            "AI Notes Generator",
            title_style
        )
    )

    story.append(Spacer(1, 0.3 * inch))

    # --------------------------------
    # Overall Summary
    # --------------------------------

    story.append(
        Paragraph(
            "Summary",
            heading_style
        )
    )

    story.append(Spacer(1, 0.15 * inch))

    story.append(
        Paragraph(
            overall_summary,
            body_style
        )
    )

    story.append(Spacer(1, 0.3 * inch))

    # --------------------------------
    # Detailed Notes
    # --------------------------------

    story.append(
        Paragraph(
            "Detailed Notes",
            heading_style
        )
    )

    story.append(Spacer(1, 0.15 * inch))

    for para in paragraphs:

        story.append(
            Paragraph(
                para,
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                0.15 * inch
            )
        )

    story.append(
        Spacer(
            1,
            0.25 * inch
        )
    )

    # --------------------------------
    # Key Points
    # --------------------------------

    story.append(
        Paragraph(
            "Key Points",
            heading_style
        )
    )

    story.append(
        Spacer(
            1,
            0.15 * inch
        )
    )

    bullet_items = []

    for point in key_points:

        bullet_items.append(

            ListItem(

                Paragraph(
                    point,
                    bullet_style
                )

            )

        )

    story.append(

        ListFlowable(
            bullet_items,
            bulletType='bullet'
        )

    )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    response = make_response(pdf)

    response.headers["Content-Type"] = "application/pdf"

    response.headers[
        "Content-Disposition"
    ] = "attachment; filename=AI_Notes.pdf"

    return response