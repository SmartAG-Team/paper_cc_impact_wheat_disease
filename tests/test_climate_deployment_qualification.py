def test_deployment_requires_every_declared_scientific_gate():
    from models.monitoring.audit_climate_calibration import deployment_decision, REQUIRED_GATES
    complete = {name: True for name in REQUIRED_GATES}
    assert deployment_decision(complete)['qualified']
    for name in REQUIRED_GATES:
        missing = dict(complete); missing.pop(name)
        answer = deployment_decision(missing)
        assert not answer['qualified'] and name in answer['failed']


def test_deployment_does_not_treat_unverified_strings_as_evidence():
    from models.monitoring.audit_climate_calibration import deployment_decision, REQUIRED_GATES
    evidence = {name: True for name in REQUIRED_GATES}
    evidence['comparable_endpoint_mapping'] = 'unknown'
    answer = deployment_decision(evidence)
    assert not answer['qualified']
    assert 'comparable_endpoint_mapping' in answer['failed']
