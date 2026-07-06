from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("customizer", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="customizationjob",
            name="scale",
            field=models.FloatField(default=1.0),
        ),
        migrations.AddField(
            model_name="customizationjob",
            name="rotation_deg",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="customizationjob",
            name="offset_x",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="customizationjob",
            name="offset_y",
            field=models.FloatField(default=0.0),
        ),
    ]
