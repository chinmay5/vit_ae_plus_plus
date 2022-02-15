from dataset.breast_cancer import pretrain_breast_cancer_clinical, pretrain_breast_cancer_data


def build_dataset_from_breast_cancer_factory(selection_type, mode, transforms, use_z_score):
    assert selection_type in ['lymph', 'clinical', 'pathology'], f"Invalid selection of type {selection_type}"
    print(f"Using selection type {selection_type}")
    if selection_type == 'lymph':
        return pretrain_breast_cancer_data.build_dataset(mode=mode, transforms=transforms, use_z_score=use_z_score)
    else:
        return pretrain_breast_cancer_clinical.build_dataset(selection_type=selection_type, mode=mode, transforms=transforms, use_z_score=use_z_score)