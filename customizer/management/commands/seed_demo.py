from pathlib import Path
import cv2
from django.core.files import File
from django.core.management.base import BaseCommand
from django.conf import settings

from customizer.models import Product, ProductImage





# Samples with no guide box (real-world photos as an admin would actually
# upload them) -- print area is specified explicitly, exactly as an admin
# would enter it in the admin panel for a brand new product photo.
MANUAL_SAMPLES = [
    # name,          slug,          view,    image path,                                    x,   y,   w,   h,  max_tilt_deg
    ("Pink Tee",       "pink-tee",       "back",  "products/pink_tee/back/base.jpg",       599, 810, 799, 900, 18.0),
    ("Maroon Tee",      "maroon-tee",     "front", "products/maroon_tee/front/base.jpg",     600, 436, 800, 634, 18.0),
    ("Two-Tone Cap",    "twotone-cap",    "side",  "products/twotone_cap/side/base.jpg",     432, 768, 720, 528, 8.0),
    # ---------------- HOODIE ----------------
(
    "Hoodie",
    "hoodie",
    "front",
    "products/hoodie/front/base.jpg",
    420,
    380,
    520,
    620,
    18.0,
),

(
    "Hoodie",
    "hoodie",
    "back",
    "products/hoodie/back/base.jpg",
    420,
    380,
    520,
    620,
    18.0,
),

# ---------------- CAP ----------------
(
    "Cap",
    "cap",
    "front",
    "products/cap/front/base.jpg",
    120,
    95,
    270,
    120,
    10.0,
),

(
    "Cap",
    "cap",
    "back",
    "products/cap/back/base.jpg",
    120,
    95,
    270,
    120,
    10.0,
),
    # ---------------- ORANGE TEE ----------------
(
    "Orange Tee",
    "orange-tee",
    "back",
    "products/orange_tee/back/orange_tee.jpg",
    300,     # X
    220,     # Y
    650,     # W
    720,     # H
    18.0,
),

# ---------------- TENNIS CAP ----------------
(
    "Tennis Cap",
    "tennis-cap",
    "front",
    "products/tennis_cap/front/otto_front.jpg",
    120,     # X
    115,     # Y
    280,     # W
    120,     # H
    10.0,
),

# ---------------- KNIT CAP ----------------
(
    "Knit Cap",
    "knit-cap",
    "front",
    "products/knit_cap/front/knitt_cap.jpg",
    90,      # X
    190,     # Y
    360,     # W
    140,     # H
    10.0,
),
]

class Command(BaseCommand):
    help = "Seed demo products."

    def handle(self, *args, **options):

        for name, slug, view, rel_path, x, y, w, h, max_tilt in MANUAL_SAMPLES:

            product, _ = Product.objects.get_or_create(
                name=name,
                slug=slug,
            )

            src = Path(settings.MEDIA_ROOT) / rel_path

            if not src.exists():
                self.stdout.write(
                    self.style.WARNING(f"Missing sample asset {src}, skipping.")
                )
                continue

            self._create(
                product,
                view,
                src,
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                },
                max_tilt_deg=max_tilt,
            )

    def _create(self, product, view, src, area, max_tilt_deg=18.0):

        pi, created = ProductImage.objects.get_or_create(
            product=product,
            view=view,
            defaults={
                "print_area_x": area["x"],
                "print_area_y": area["y"],
                "print_area_w": area["w"],
                "print_area_h": area["h"],
                "max_tilt_deg": max_tilt_deg,
            },
        )

        if created:
            with open(src, "rb") as f:
                pi.base_image.save(src.name, File(f), save=True)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Created {product.name} / {view}"
                )
            )

        else:
            self.stdout.write(
                f"Already exists: {product.name} / {view}"
            )