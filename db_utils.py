from typing import Annotated
from datetime import datetime

from fastapi import Depends
from sqlmodel import Field, Session, SQLModel, create_engine, select


class DocumentStore(SQLModel,table=True):
    id: int = Field(default=None, primary_key=True)
    filename: str
    uploaded_at : datetime = Field(default_factory=datetime.now)


DATABASE_URL = "postgresql+psycopg2://username:password@host/dbname"

engine = create_engine(DATABASE_URL)

# create the db and tables
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# get the session for performing database operations
def get_session():
    with Session(engine) as session:
        yield session

# for dependency injection
SessionDep = Annotated[Session,Depends(get_session)]


#add document
def add_document(document_store: DocumentStore, session: Session):
    session.add(document_store)
    session.commit()


#list document
def list_documents(session:Session)->list[DocumentStore]:
    return session.exec(select(DocumentStore)).all()


# delete document
def delete_document(document_id: int, session:Session):
    document = session.get(DocumentStore,document_id)
    if not document:
        raise ValueError(f"Document with ID {document_id} not found.")
    session.delete(document)
    session.commit()

create_db_and_tables()