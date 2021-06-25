import torch
from tqdm import tqdm
from transmatching.Model.model import Model
from argparse import ArgumentParser
from transmatching.Utils.utils import get_errors, area_weighted_normalization, chamfer_loss, approximate_geodesic_distances
import numpy as np
from pytorch_lightning import seed_everything
from scipy.io import loadmat


def main(args):
# ------------------------------------------------------------------------------------------------------------------
# BEGIN SETUP  -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    seed_everything(0)

    faust = loadmat(args.path_data)
    shapes = faust["vertices"]
    faces = faust["faces"] - 1
    n_pairs = 100
    n = shapes.shape[0]

    # INITIALIZE MODEL
    model = Model(args.d_bottleneck, args.d_latent, args.d_channels, args.d_middle, args.N,
                  args.heads, args.max_seq_len, args.d_origin, args.dropout).cuda()

    model.load_state_dict(torch.load("models/"+args.run_name))
    print("MODEL RESUMED ---------------------------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------------------------------------------
# END SETUP  -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        err = []
        for _ in tqdm(range(n_pairs)):

            shape_A_idx = np.random.randint(n)

            shape_B_idx = np.random.randint(n)
            while shape_A_idx == shape_B_idx:  # avoid taking A exactly equal to B
                shape_B_idx = np.random.randint(n)

            shape_A = shapes[shape_A_idx]
            shape_B = shapes[shape_B_idx]

            geod = approximate_geodesic_distances(shape_B, faces.astype("int"))
            geod /= np.max(geod)

            points_A = area_weighted_normalization(torch.from_numpy(shape_A), rescale=False)
            points_B = area_weighted_normalization(torch.from_numpy(shape_B), rescale=False)

            y_hat_1 = model(points_A.unsqueeze(0).float().cuda(), points_B.unsqueeze(0).float().cuda())
            y_hat_2 = model(points_B.unsqueeze(0).float().cuda(), points_A.unsqueeze(0).float().cuda())

            d12 = chamfer_loss(points_A.float().cuda(), y_hat_1)
            d21 = chamfer_loss(points_B.float().cuda(), y_hat_2)

            if d12 < d21:
                d = torch.cdist(points_A.float().cuda(), y_hat_1).squeeze(0).cpu()
                err.extend(get_errors(d, geod))
            else:
                d = torch.cdist(points_B.float().cuda(), y_hat_2).squeeze(0).cpu()
                err.extend(get_errors(d.transpose(1, 0), geod))

        print("ERROR: ", np.mean(np.array(err)))


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--run_name", default="trained_model")

    parser.add_argument("--d_bottleneck", default=32)
    parser.add_argument("--d_latent", default=64)
    parser.add_argument("--d_channels", default=64)
    parser.add_argument("--d_origin", default=3)
    parser.add_argument("--d_middle", default=512)

    parser.add_argument("--N", default=8)
    parser.add_argument("--heads", default=4)
    parser.add_argument("--max_seq_len", default=100)

    parser.add_argument("--dropout", default=0.00)
    parser.add_argument("--num_workers", default=0)

    parser.add_argument("--path_data", default="dataset/FAUST_noise_0.00.mat")

    args = parser.parse_args()

    main(args)






























