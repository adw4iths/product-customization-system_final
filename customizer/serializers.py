from rest_framework import serializers
from .models import Product, ProductImage, DesignUpload, CustomizationJob


class ProductImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductImage
        fields = ["id", "view", "base_image", "print_area_x", "print_area_y", "print_area_w", "print_area_h"]


class ProductSerializer(serializers.ModelSerializer):
    images = ProductImageSerializer(many=True, read_only=True)

    class Meta:
        model = Product
        fields = ["id", "name", "slug", "images"]


class DesignUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = DesignUpload
        fields = ["id", "file", "uploaded_at"]
        read_only_fields = ["id", "uploaded_at"]


class CustomizationJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomizationJob
        fields = [
            "id", "design", "product_image", "status", "result_image", "error_message",
            "scale", "rotation_deg", "offset_x", "offset_y",
            "created_at", "finished_at",
        ]
        read_only_fields = ["id", "status", "result_image", "error_message", "created_at", "finished_at"]


class CreateCustomizationJobSerializer(serializers.Serializer):
    design_id = serializers.UUIDField()
    product_image_id = serializers.IntegerField()

    # Manual placement from the interactive preview editor. All optional --
    # omitting them (or the whole request just sending design_id +
    # product_image_id, as the original single-page demo does) reproduces
    # the original auto-centered, auto-fit placement exactly.
    scale = serializers.FloatField(required=False, default=1.0, min_value=0.1, max_value=4.0)
    rotation_deg = serializers.FloatField(required=False, default=0.0, min_value=-180.0, max_value=180.0)
    offset_x = serializers.FloatField(required=False, default=0.0, min_value=-0.5, max_value=0.5)
    offset_y = serializers.FloatField(required=False, default=0.0, min_value=-0.5, max_value=0.5)
