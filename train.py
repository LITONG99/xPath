import torch
import numpy as np
import copy
from config import GPU, TARGET_NTYPE, HGN_TYPE, N_LAYER, NUM_CLASSES, hgn_path, graph_path, index_path, logger
from utils import get_model, accuracy


if __name__ == '__main__':
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    g, model, _info = get_model(HGN_TYPE, N_LAYER, NUM_CLASSES, graph_path, index_path)

    model.to(device)
    model.g = g.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
    train_node = _info["train_index"].long().to(device)
    train_label = _info["train_label"].long().to(device)
    valid_node = _info["valid_index"].long().to(device)
    valid_label = _info["valid_label"].long().to(device)
    test_node = _info["test_index"].long().to(device)
    test_label = _info["test_label"].long().to(device)

    patience = 0
    best_score = 0
    max_score = 0
    min_loss = np.inf
    epoch = 0
    log_epoch = 100
    max_epoch = 2000
    max_patience = 50

    x = model.g.ndata.pop("nfeat")
    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(x, TARGET_NTYPE, train_node, train_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        if epoch % log_epoch == 0:
            logger.info(f"Epoch: {epoch}\t Loss: {loss:.4f}")

        # Validation
        model.eval()
        logits = model.forward(x, TARGET_NTYPE)
        train_acc = accuracy(logits[train_node], train_label)
        train_loss = model.cross_entropy_loss(logits[train_node], train_label).cpu().item()
        val_acc = accuracy(logits[valid_node], valid_label)
        val_loss = model.cross_entropy_loss(logits[valid_node], valid_label).cpu().item()
        if epoch % log_epoch == 0:
            logger.info(f"Train: {train_acc:.3f}, {train_loss:.3f}, Val: {val_acc:.3f}, {val_loss:.3f}")
        if val_loss <= min_loss or val_acc >= max_score:
            if val_acc >= best_score:
                best_score = val_acc
                best_model = copy.deepcopy(model.state_dict())
            min_loss = np.min((min_loss, val_loss))
            max_score = np.max((max_score, val_acc))
            patience = 0
        else:
            patience += 1
            if patience == max_patience:
                model.load_state_dict(best_model)
                break

    # Test
    model.eval()
    logits = model.forward(x, TARGET_NTYPE)
    test_acc = accuracy(logits[test_node], test_label)
    logger.info(f"Test ACC = {test_acc}")

    torch.save(
        {
            "epoch": epoch,
            "model_type": HGN_TYPE,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        hgn_path,
    )