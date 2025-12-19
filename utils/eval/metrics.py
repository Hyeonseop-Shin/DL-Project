
import torch

eps = 1e-12


def RSE(pred, true):
    num = torch.sqrt(torch.sum((true - pred) ** 2))
    den = torch.sqrt(torch.sum((true - true.mean()) ** 2))
    return num / (den + eps)


def CORR(pred, true):
    true_mean = true.mean(dim=0)
    pred_mean = pred.mean(dim=0)

    true_centered = true - true_mean
    pred_centered = pred - pred_mean

    numerator = (true_centered * pred_centered).sum(dim=0)
    denominator = torch.sqrt((true_centered ** 2).sum(dim=0)) * \
                  torch.sqrt((pred_centered ** 2).sum(dim=0))

    corr = numerator / (denominator + eps)
    return corr.mean()


def MAE(pred, true):
    return torch.mean(torch.abs(true - pred))


def MSE(pred, true):
    return torch.mean((true - pred) ** 2)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return torch.mean(torch.abs((true - pred) / (true + eps)))


def MSPE(pred, true):
    return torch.mean(((true - pred) / (true + eps)) ** 2)


def metric(pred, true):
    mae = MAE(pred, true).detach().cpu().numpy()
    mse = MSE(pred, true).detach().cpu().numpy()
    rmse = RMSE(pred, true).detach().cpu().numpy()
    mape = MAPE(pred, true).detach().cpu().numpy()
    mspe = MSPE(pred, true).detach().cpu().numpy()
    corr = CORR(pred, true).detach().cpu().numpy()
    return mae, mse, rmse, mape, mspe, corr
