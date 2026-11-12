"""Index reported paper citations without upgrading them to local evidence."""
from .core import digest,validate_method_sources


def graph_records(method,method_id,task,protocol,revision):
    from .runtime import node,edge
    if method['origin']!='paper_claim':return [],[]
    validate_method_sources(method)
    nodes=[];edges=[]
    for citation in dict.fromkeys(method['sources']):
        source='source-'+digest([task,protocol,citation])
        nodes.append(node(source,'Source',{'citation':citation,'citation_hash':digest(citation),
            'verification_status':'agent_reported_unverified','source_version':'unknown',
            'retrieved_content_ref':None,
            'qualification':'Reported citation, not independently verified content, local experiment or mechanism evidence.'},task,protocol,'paper_claim'))
        relationship=edge(method_id+'-source-'+digest(citation)[:16],method_id,source,'DERIVED_FROM',[source],task,revision)
        relationship['attrs'].update(origin='paper_claim',status='provisional')
        edges.append(relationship)
    return nodes,edges
