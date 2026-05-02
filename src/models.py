"""
MongoEngine models for document extraction
"""
from mongoengine import (
    Document,
    StringField,
    IntField,
    DictField,
    DateTimeField,
    FloatField,
    ListField,
    EmbeddedDocument,
    EmbeddedDocumentField
)
from datetime import datetime
from typing import Optional, Dict, Any


class ExtractionResult(EmbeddedDocument):
    """Embedded document for storing extraction results from Gemini"""
    category = StringField(required=False)
    main_topic = StringField(required=False)
    key_entities = ListField(StringField(), default=list)
    language = StringField(required=False)
    confidence = FloatField(required=False)
    summary = StringField(required=False)

    # Additional fields for flexible data storage
    additional_data = DictField(default=dict)

    meta = {
        'strict': False  # Allow flexible schema
    }


class DocumentExtraction(Document):
    """
    Main document model for storing document extraction results
    """
    # File metadata
    filename = StringField(required=True, max_length=255)
    content_type = StringField(required=True, max_length=100)
    size_bytes = IntField(required=True, min_value=0)

    # Extraction results
    extraction_result = EmbeddedDocumentField(ExtractionResult)

    # Additional metadata
    metadata = DictField(default=dict)

    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow, required=True)
    updated_at = DateTimeField(default=datetime.utcnow, required=True)

    # Indexes
    meta = {
        'collection': 'documents',
        'indexes': [
            'filename',
            'content_type',
            '-created_at',  # Descending index for latest first
            {
                'fields': ['created_at'],
                'expireAfterSeconds': None  # Set TTL if needed
            }
        ],
        'strict': False,  # Allow flexible schema
        'ordering': ['-created_at']  # Default ordering
    }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary for API responses

        Returns:
            Dictionary representation of the document
        """
        result = {
            'id': str(self.id),
            'filename': self.filename,
            'content_type': self.content_type,
            'size_bytes': self.size_bytes,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

        # Convert extraction result
        if self.extraction_result:
            result['extraction_result'] = {
                'category': self.extraction_result.category,
                'main_topic': self.extraction_result.main_topic,
                'key_entities': self.extraction_result.key_entities,
                'language': self.extraction_result.language,
                'confidence': self.extraction_result.confidence,
                'summary': self.extraction_result.summary,
                'additional_data': self.extraction_result.additional_data
            }

        return result

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.utcnow()
        self.save()

    def __str__(self):
        return f"DocumentExtraction(id={self.id}, filename={self.filename})"

    def __repr__(self):
        return self.__str__()
