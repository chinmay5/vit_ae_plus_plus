# Call the ssl feature extraction module
import os
import pickle

from environment_setup import PROJECT_ROOT_DIR


def generate_modules():
    global model, args
    Folder = 'k_fold_brats'
    ROOT_DIR = f'/home/chinmayp/workspace/vit_3d/output_dir/{Folder}/checkpoints'
    recursive_term = '/checkpoints'
    for idx in range(3):
        DIR = ROOT_DIR + idx * recursive_term
        all_models = os.listdir(DIR)
        checkpoints = list(
            filter(lambda x: 'checkpoint-' in x and 'min_loss' not in x and 'k_fold' not in x, all_models))
        # For each of the checkpoiints, we should generate the features
        import fine_tune.extract_ssl_features as feat_extr
        args = feat_extr.get_args_parser()
        args = args.parse_args()
        for model in checkpoints:
            # We change a few parameters here since we need to extract features for each of the modules
            external_config_injection = {
                'feature_extractor_load_path': f'output_dir/{Folder}{idx * recursive_term}',
                'output_dir': f'output_dir/{Folder}/{model}',
                'subtype': f'acc_vs_epoch' + f'{idx * recursive_term}' + f'/{model}',
                'checkpoint': model
            }
            feat_extr.main(args, external_config_injection=external_config_injection)


def get_numbers():
    recursive_term = '/checkpoints'
    for idx in range(3):
        candidates = os.listdir(
            os.path.join(os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', f'acc_vs_epoch/{recursive_term*idx}')))
        result_dict = {}
        # Filter away the folders
        candidates = [x for x in candidates if 'checkpoint-' in x]
        import ablation.radiomics_k_fold as rko
        for filename in candidates:
            rko.FILENAME = f'acc_vs_epoch/{filename}/{recursive_term*idx}'
            spec_list, sens_list = rko.evaluate_features(is_50=False)
            result_dict[filename] = (spec_list, sens_list)
        pickle.dump(result_dict, open(f'result_{idx}.pkl', 'wb'))


def plot_results():
    result_dict = pickle.load(open('result.pkl', 'rb'))
    sort_keys = sorted(result_dict, key=lambda x: int(x[x.find("-") + 1: x.find(".pth")]))
    for key in sort_keys:
        result_dict[key]


if __name__ == '__main__':
    # generate_modules()
    get_numbers()
