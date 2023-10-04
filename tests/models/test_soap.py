import numpy as np
import torch
from dscribe.descriptors.soap import SOAP
from pytest import mark, fixture

from jitterbug.model.soap import make_gpr_model, train_model


@fixture
def descriptors(train_set):
    species = sorted(set(sum([a.get_chemical_symbols() for a in train_set], [])))
    soap = SOAP(
        species=species,
        r_cut=4.,
        n_max=4,
        l_max=4,
    )
    return soap.create(train_set)


@mark.parametrize('use_adr', [True, False])
def test_make_model(use_adr, descriptors):
    model = make_gpr_model(descriptors, 4, use_adr_kernel=use_adr)

    # Evaluate on a single point
    model.eval()
    model.likelihood.eval()
    pred_dist = model(torch.from_numpy(descriptors[0, :, :]))
    assert torch.isclose(pred_dist.mean, torch.zeros((1, 3,), dtype=pred_dist.mean.dtype)).all(), [pred_dist.mean]


def test_train(descriptors, train_set):
    # Make the model and the training set
    train_y = np.array([a.get_potential_energy() for a in train_set])
    train_y -= train_y.min()
    model = make_gpr_model(descriptors, 32)

    # Evaluate the untrained model
    model.eval()
    pred_y = model(torch.from_numpy(descriptors))
    error_y = pred_y.mean.sum(dim=1).detach().numpy() - train_y
    mae_untrained = np.abs(error_y).mean()

    # Train
    losses = train_model(model, descriptors, train_y, 16)
    assert len(losses) == 16

    # Run the evaluation
    model.eval()
    pred_y = model(torch.from_numpy(descriptors))
    error_y = pred_y.mean.sum(dim=1).detach().numpy() - train_y
    mae_trained = np.abs(error_y).mean()
    assert mae_trained < mae_untrained
