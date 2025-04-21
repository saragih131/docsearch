from django.db import models
from django.utils import timezone

class Document(models.Model):
    id = models.AutoField(primary_key=True)
    document_name = models.CharField(max_length=500)
    document_uploaddate = models.DateField(default=timezone.now)
    document_path = models.CharField(max_length=500)

    class Meta:
        db_table = 'documents'
        ordering = ['id']

    def __str__(self):
        return self.document_name

class Result(models.Model):
    id = models.AutoField(primary_key=True)
    judul_artikel = models.CharField(max_length=500)
    ekstrak_teks = models.TextField()
    tokenisasi = models.TextField()
    kueri_ocr = models.TextField()
    pelatihan_model = models.TextField()
    perhitungan_vektor_dokumen = models.TextField()
    perhitungan_kueri_ocr = models.TextField()
    perhitungan_kesamaan_kosinus = models.FloatField()
    keterangan = models.CharField(max_length=100)

    class Meta:
        db_table = 'results'
        ordering = ['id']

    def __str__(self):
        return f"Result {self.id}"