import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transmatching.Data.dataset_smpl import SMPLDataset
from transmatching.Model.model import Model
from argparse import ArgumentParser


def main(args):

# ------------------------------------------------------------------------------------------------------------------
# BEGIN SETUP  -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------


    # DATASET
    data_train = SMPLDataset(args.path_data, train=True)

    # DATALOADERS
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)

    # INITIALIZE MODEL
    model = Model(args.d_bottleneck, args.d_latent, args.d_channels, args.d_middle,
                  args.N, args.heads, args.max_seq_len, args.d_origin, args.dropout).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-9)

# ------------------------------------------------------------------------------------------------------------------
# END SETUP  -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# BEGIN TRAINING ---------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    print("TRAINING --------------------------------------------------------------------------------------------------")
    start = time.time()
    for epoch in range(args.n_epoch):
        model = model.train()
        ep_loss = 0
        for item in tqdm(dataloader_train):

            shapes = item["x"].cuda()
            shape1 = shapes[:args.batch_size // 2, :, :]
            shape2 = shapes[args.batch_size // 2:, :, :]

            y_hat = model(shape1, shape2)
            loss = ((y_hat - shape1) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

        print(f"EPOCH: {epoch} HAS FINISHED, in {time.time() - start} SECONDS! ---------------------------------------")
        start = time.time()
        print(f"LOSS: {ep_loss} --------------------------------------------------------------------------------------")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/"+args.run_name)


# ------------------------------------------------------------------------------------------------------------------
# END TRAINING -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--run_name", default="custom_train")

    parser.add_argument("--d_bottleneck", default=32)
    parser.add_argument("--d_latent", default=64)
    parser.add_argument("--d_channels", default=64)
    parser.add_argument("--d_origin", default=3)
    parser.add_argument("--d_middle", default=512)

    parser.add_argument("--N", default=8)
    parser.add_argument("--heads", default=4)
    parser.add_argument("--max_seq_len", default=100)

    parser.add_argument("--dropout", default=0.01)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--n_epoch", default=1000000)
    parser.add_argument("--batch_size", default=16)

    parser.add_argument("--num_workers", default=8)

    parser.add_argument("--path_data", default="dataset/")

    args = parser.parse_args()

    main(args)






























