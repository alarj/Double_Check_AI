import lxml.etree as ET

def parse_procurement_xml(file_path):
    """Võtab XML-ist välja olulised faktid."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {
            'cbc': 'urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2',
            'cac': 'urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2'
        }
        results = []
        # Otsime hankeid
        for project in root.xpath('.//cac:ProcurementProject', namespaces=ns):
            name = project.xpath('./cbc:Name/text()', namespaces=ns)
            desc = project.xpath('./cbc:Description/text()', namespaces=ns)
            content = f"HANKE INFO: {name[0] if name else 'Nimetu'}. {desc[0] if desc else ''}"
            results.append({
                "content": content,
                "metadata": {"source": file_path.split('/')[-1]}
            })
        return results
    except Exception as e:
        print(f"Viga: {e}")
        return []