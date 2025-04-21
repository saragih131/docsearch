from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('search/', views.search_documents, name='search_documents'),
    path('add_document/', views.add_document, name='add_document'),
    path('delete_document/<int:document_id>/', views.delete_document, name='delete_document'),
    path('download_document/<int:document_id>/', views.download_document, name='download_document'),
]

