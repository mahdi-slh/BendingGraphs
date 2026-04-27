import glob
from os import path
from trainer import trainer
from configs import *
# import evaluation.eval_match as eval_rigid
import evaluation.test_match as test_match
# import evaluation.eval_deform as eval_deform
from evaluation.eval_rigid import EvaluatorRigid
from utils.utils import (
    TripletSampler,
    MaskedSampler,
    get_model_path,
    create_triplet_validation_masks,
    create_validation_masks,
)
from models.model import init_model

# import evaluation.evaluatemodelnet
import evaluation.save_for_fmnet as save_for_fmnet
from torch_geometric.data import DataLoader

# from torch.utils.data import DataLoader as UtilsDataLoader
from torch.utils.data import BatchSampler


from loaders.modelnet_dataset import ModelNetDataset
from loaders.body_dataset import BodyDataset
from loaders.syn_prim_dataset import SynPrimDataset
from loaders.match_3d_dataset import Match3DDataset

# from loaders.faust_dataset import FaustDataset
from loaders.smal_dataset import SMALDataset
from loaders.tosca_dataset import ToscaDataset
from loaders.faust_syn_dataset import FaustSynDataset
from loaders.surreal_dataset import SURREALDataset
from loaders.SHREC19_dataset import SHREC19Dataset
from loaders.modelnet_mesh_dataset import ModelNetMeshDataset
from loaders.shrec_partial_dataset import ShrecPartialDataset

from torch.utils.tensorboard import SummaryWriter
from utils.checks import check_valid


def routine_modelnet(stage="auto", load_state=True):
    settings = training_configs()
    settings.input_params()

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    if path.exists(trained_path) and stage == "auto":

        print("Evaluation on ModelNet...")

        modelnet_dataset_test = ModelNetDataset(
            MODELNET_DIR, train=False, mode=settings.data_mode
        )
        for e in range(18, 19):
            model = init_model(get_model_path(False, settings), settings)
            eval_rigid.test(settings, model, modelnet_dataset_test)

        evaluation.evaluate_model(model, settings, modelnet_dataset_test, True)

    # train
    elif (path.exists(pretrained_path) and stage == "auto") or stage == "main":

        print("Training on ModelNet...")

        modelnet_dataset_train = ModelNetDataset(
            MODELNET_DIR, train=True, mode=settings.data_mode
        )
        modelnet_dataset_test = ModelNetDataset(
            MODELNET_DIR, train=False, mode=settings.data_mode
        )
        train_mask, val_mask = create_triplet_validation_masks(
            modelnet_dataset_train, VALIDATION_SPLIT
        )
        modelnet_val_loader = DataLoader(
            modelnet_dataset_train,
            batch_sampler=BatchSampler(
                TripletSampler(len(modelnet_dataset_train) - 1, val_mask),
                batch_size=settings.mb_size,
                drop_last=True,
            ),
        )
        modelnet_train_loader = DataLoader(
            modelnet_dataset_train,
            batch_sampler=BatchSampler(
                TripletSampler(len(modelnet_dataset_train) - 1, train_mask),
                batch_size=settings.mb_size,
                drop_last=True,
            ),
        )
        # modelnet_test_loader = DataLoader(modelnet_dataset_test, batch_size=settings.mb_size)

        feat_trainer = trainer(
            settings,
            modelnet_train_loader,
            modelnet_val_loader,
            dataset_test=modelnet_dataset_test,
            eval_type=eval_rigid.test,
            pretrain=False,
            model_path=pretrained_path,
        )
        feat_trainer.train()

    # pretrain
    else:

        print("Training on Synthetic Primitves...")

        synprim_sub_dirs = [1]
        # if VERSION == 2:
        synprim_dataset_train = SynPrimDataset(
            root=SYN_PRIM_DIR, dirs=synprim_sub_dirs, train_val=True
        )
        synprim_dataset_val = SynPrimDataset(
            root=SYN_PRIM_DIR, dirs=synprim_sub_dirs, train_val=False
        )
        syn_val_loader = DataLoader(
            synprim_dataset_train, batch_size=settings.mb_size
        )
        syn_train_loader = DataLoader(
            synprim_dataset_val, batch_size=settings.mb_size
        )
        # else:
        #     synprim_dataset = SynPrimDataset(root=SYN_PRIM_DIR, dirs=synprim_sub_dirs)
        #     train_mask, val_mask = create_triplet_validation_masks(
        #         synprim_dataset, VALIDATION_SPLIT
        #     )
        #     syn_val_loader = DataLoader(
        #         synprim_dataset,
        #         batch_sampler=BatchSampler(
        #             TripletSampler(len(synprim_dataset) - 1, val_mask),
        #             batch_size=settings.mb_size,
        #             drop_last=True,
        #         ),
        #     )
        #     syn_train_loader = DataLoader(
        #         synprim_dataset,
        #         batch_sampler=BatchSampler(
        #             TripletSampler(len(synprim_dataset) - 1, train_mask),
        #             batch_size=settings.mb_size,
        #             drop_last=True,
        #         ),
        #     )

        feat_trainer = trainer(
            settings, syn_train_loader, syn_val_loader, dataset_test=None, pretrain=True
        )
        feat_trainer.train()

        print("Training on ModelNet...")

        modelnet_dataset_train = ModelNetDataset(
            MODELNET_DIR, train=True, mode=settings.data_mode
        )
        modelnet_dataset_test = ModelNetDataset(
            MODELNET_DIR, train=False, mode=settings.data_mode
        )
        train_mask, val_mask = create_triplet_validation_masks(
            modelnet_dataset_train, VALIDATION_SPLIT
        )
        modelnet_val_loader = DataLoader(
            modelnet_dataset_train,
            batch_sampler=BatchSampler(
                TripletSampler(len(modelnet_dataset_train) - 1, val_mask),
                batch_size=settings.mb_size,
                drop_last=True,
            ),
        )
        modelnet_train_loader = DataLoader(
            modelnet_dataset_train,
            batch_sampler=BatchSampler(
                TripletSampler(len(modelnet_dataset_train) - 1, train_mask),
                batch_size=settings.mb_size,
                drop_last=True,
            ),
        )
        # modelnet_test_loader = DataLoader(modelnet_dataset_test, batch_size=settings.mb_size)
        feat_trainer = trainer(
            settings,
            modelnet_train_loader,
            modelnet_val_loader,
            dataset_test=modelnet_dataset_test,
            eval_type='rigid',
            pretrain=False,
            model_path=pretrained_path,
        )
        feat_trainer.train()


def routine_3dmatch(stage="auto", load_state=True):

    settings = training_configs()
    settings.input_params()

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    if path.exists(trained_path) or stage == "eval":

        print("Evaluation on 3D Match...")
        match3d_dataset_test = Match3DDataset(MATCH3D_DIR)
        for e in range(18, 19):
            model = init_model(get_model_path(False, settings), settings)
            test_match.test(settings, model, match3d_dataset_test)
    else:

        print(
            "Routine only evaluates on 3Dmatch, please train on modelnet first or use pretrained models"
        )


def routine_faust(stage="auto", load_state=True):
    settings = training_configs()
    settings.dataset = "faust"
    settings.input_params()

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)
    # trained_path = get_model_name(False, settings,epoch='29')
    print("found pretrained:", path.exists(pretrained_path), pretrained_path)
    print("found trained:", path.exists(trained_path), trained_path)
    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":

        # print("Saving for FMNet...")

        # root='../../../dataset/MPI-FAUST/'
        # mpi_dataset=FaustDataset(root,True)
        # # test_loader = DataLoader(mpi_dataset, batch_size=32)
        # model = init_model(get_model_name(False, settings), settings)
        # save_for_fmnet.test(settings, model, mpi_dataset)

        print("Evaluating on FAUST...")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR, train=False, mode=settings.data_mode
        )


        eval_deform.test(settings, model, faust_dataset_test, full_eval=True)

    # train
    # elif (path.exists(pretrained_path) and stage == 'auto') or stage == 'main':

    else:

        print("Training on MPI-FAUST...")
        faust_dataset_trainval = FaustSynDataset(
            root=MPIFAUST_DIR, train=True, mode=settings.data_mode
        )
        train_mask, val_mask = create_validation_masks(
            faust_dataset_trainval, VALIDATION_SPLIT
        )
        # faust_dataset_test = faust_dataset_trainval
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR, train=False, mode=settings.data_mode
        )
        # test_deform.test(settings,model,faust_dataset_test)
        faust_val_loader = DataLoader(
            faust_dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(faust_dataset_trainval), val_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )
        faust_loader_train = DataLoader(
            faust_dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(faust_dataset_trainval), train_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )

        # model = init_model(get_model_name(False, settings), settings)
        # test_deform.test(settings,model,faust_dataset_test)
        # trained_path = get_model_name(False, settings,epoch='49')
        # trained_path = '../model/trained/_faustnoisea10m1lr0.001f64r7e29.model'
        feat_trainer = trainer(
            settings,
            faust_loader_train,
            faust_val_loader,
            dataset_test=faust_dataset_test,
            eval_type=eval_deform.test,
            pretrain=False,
            model_path=trained_path,
        )
        feat_trainer.train()


def routine_modelnet_mesh(stage="auto", load_state=True):
    settings = training_configs()
    settings.dataset = "modelnetmesh"

    settings.input_params()
    # settings.identifier='SFGOT1GRU2'

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    # init_model(trained_path, settings)

    # trained_path = get_model_name(False, settings,epoch='29')
    print("found pretrained:", path.exists(pretrained_path), pretrained_path)
    print("found trained:", path.exists(trained_path), trained_path)
    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":
        print("Evaluating on ModelNet...")
        model, _, e = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        # modelnet_dataset_test = ModelNetMeshDataset(
        #     root=MODELNET_DIR, train=False, mode=settings.data_mode
        # )
        modelnet_dataset_test = ModelNetMeshDataset(
            root=MODELNET_DIR, train=False, mode='category'
        )
        evaluator = EvaluatorRigid(None,visualize=False)
        evaluator.eval(test_data=modelnet_dataset_test,net=model,epoch = e)

    else:
        print("Training on ModelNet...")
        
        modelnet_dataset_trainval = ModelNetMeshDataset(
            MODELNET_DIR, train=True, mode=settings.data_mode
        )
        train_mask, val_mask = create_validation_masks(
            modelnet_dataset_trainval, VALIDATION_SPLIT
        )
        # faust_dataset_test = faust_dataset_trainval
        modelnet_dataset_test = ModelNetMeshDataset(
            root=MODELNET_DIR, train=False, sample_per_frame=1, mode='clean'
        )
        # test_deform.test(settings,model,faust_dataset_test)
        modelnet_val_loader = DataLoader(
            modelnet_dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(modelnet_dataset_trainval), val_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )
        modelnet_loader_train = DataLoader(
            modelnet_dataset_trainval,
            num_workers=0,
            batch_sampler=BatchSampler(
                MaskedSampler(len(modelnet_dataset_trainval), train_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )

        
        # model = init_model(get_model_name(False, settings), settings)
        # test_deform.test(settings,model,faust_dataset_test)
        # trained_path = get_model_name(False, settings,epoch='49')
        # trained_path = '../model/trained/_faustnoisea10m1lr0.001f64r7e29.model'
        feat_trainer = trainer(
            settings,
            modelnet_loader_train,
            modelnet_val_loader,
            dataset_test=modelnet_dataset_test,
            surface_type='rigid',
            pretrain=False,
            model_path=trained_path,
        )
        feat_trainer.train()


def routine_surreal_faust(stage="auto", load_state=True):
    settings = training_configs()
    settings.dataset = "surreal"
    settings.input_params()
    # settings.identifier='s3knognn'
    # settings.identifier='SFGOT1GRU2BCE4'

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    # init_model(trained_path, settings)

    # trained_path = get_model_name(False, settings,epoch='29')
    print("found pretrained:", path.exists(pretrained_path), pretrained_path)
    print("found trained:", path.exists(trained_path), trained_path)
    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":
        print("Evaluating on FAUST...")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR, train=False, mode=settings.data_mode
        )

        eval_deform.test(
            settings, model, faust_dataset_test, full_eval=True, visualize=True
        )

    # train
    # elif (path.exists(pretrained_path) and stage == 'auto') or stage == 'main':

    elif path.exists(trained_path) and stage == "eval_metrics":
        print("Evaluating on FAUST metrics... This would take ages..")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR, train=False, mode=settings.data_mode
        )

        eval_deform.test(
            settings, model, faust_dataset_test, full_eval=True, visualize=False
        )
    else:
        print("Training on SURREAL...")
        faust_dataset_trainval = SURREALDataset(
            SURREAL_DATA_DIR, SURREAL_MODEL_DIR, train=True, mode=settings.data_mode
        )
        train_mask, val_mask = create_validation_masks(
            faust_dataset_trainval, VALIDATION_SPLIT
        )
        # faust_dataset_test = faust_dataset_trainval
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR,
            train=False,
            frame_range=range(90, 99),
            sample_per_frame=1,
            mode=settings.data_mode,
        )
        # test_deform.test(settings,model,faust_dataset_test)
        faust_val_loader = DataLoader(
            faust_dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(faust_dataset_trainval), val_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )
        faust_loader_train = DataLoader(
            faust_dataset_trainval,
            num_workers=0,
            batch_sampler=BatchSampler(
                MaskedSampler(len(faust_dataset_trainval), train_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )

        # model = init_model(get_model_name(False, settings), settings)
        # test_deform.test(settings,model,faust_dataset_test)
        # trained_path = get_model_name(False, settings,epoch='49')
        # trained_path = '../model/trained/_faustnoisea10m1lr0.001f64r7e29.model'
        feat_trainer = trainer(
            settings,
            faust_loader_train,
            faust_val_loader,
            dataset_test=faust_dataset_test,
            eval_type=eval_deform.test,
            pretrain=False,
            model_path=trained_path,
        )
        feat_trainer.train()


def routine_tosca_faust(stage="auto", load_state=True):
    settings = training_configs()
    settings.dataset = "tosca"
    settings.input_params()

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    # init_model(trained_path, settings)
    target_shape = "random"

    # trained_path = get_model_name(False, settings,epoch='29')
    print("found pretrained:", path.exists(pretrained_path), pretrained_path)
    print("found trained:", path.exists(trained_path), trained_path)
    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":
        print("Evaluating on FAUST...")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR, train=False, mode=settings.data_mode
        )

        eval_deform.test(
            settings, model, faust_dataset_test, full_eval=True, visualize=True
        )

    # train
    # elif (path.exists(pretrained_path) and stage == 'auto') or stage == 'main':

    elif path.exists(trained_path) and stage == "eval_metrics":
        print("Evaluating on FAUST metrics... This would take ages..")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR, train=False, mode=settings.data_mode
        )

        eval_deform.test(
            settings, model, faust_dataset_test, full_eval=True, visualize=False
        )
    else:
        print("Training on TOSCA...")
        faust_dataset_trainval = ToscaDataset(
            root=TOSCA_DIR,
            train=True,
            mode=settings.data_mode,
            target_shape=target_shape,
        )
        # faust_dataset_trainval = SURREALDataset(
        #     SURREAL_DATA_DIR, SURREAL_MODEL_DIR, train=True, mode=settings.data_mode)
        train_mask, val_mask = create_validation_masks(
            faust_dataset_trainval, VALIDATION_SPLIT
        )
        # faust_dataset_test = faust_dataset_trainval
        faust_dataset_test = FaustSynDataset(
            root=MPIFAUST_DIR,
            train=False,
            frame_range=range(90, 99),
            sample_per_frame=1,
            mode=settings.data_mode,
        )
        # test_deform.test(settings,model,faust_dataset_test)
        faust_val_loader = DataLoader(
            faust_dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(faust_dataset_trainval), val_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )
        faust_loader_train = DataLoader(
            faust_dataset_trainval,
            num_workers=0,
            batch_sampler=BatchSampler(
                MaskedSampler(len(faust_dataset_trainval), train_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )

        # model = init_model(get_model_name(False, settings), settings)
        # test_deform.test(settings,model,faust_dataset_test)
        # trained_path = get_model_name(False, settings,epoch='49')
        # trained_path = '../model/trained/_faustnoisea10m1lr0.001f64r7e29.model'
        feat_trainer = trainer(
            settings,
            faust_loader_train,
            faust_val_loader,
            dataset_test=faust_dataset_test,
            eval_type=eval_deform.test,
            pretrain=False,
            model_path=trained_path,
        )
        feat_trainer.train()


def routine_surreal_shrec19(stage="auto", load_state=True):
    settings = training_configs()
    settings.dataset = "surreal"
    settings.input_params()
    # settings.identifier='exp2'

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    # init_model(trained_path, settings)

    # trained_path = get_model_name(False, settings,epoch='29')
    print("found pretrained:", path.exists(pretrained_path), pretrained_path)
    print("found trained:", path.exists(trained_path), trained_path)
    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":
        print("Evaluating on SHREC19...")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        faust_dataset_test = SHREC19Dataset(
            root=SHREC19_DIR, train=False, mode=settings.data_mode
        )
        eval_deform.test(
            settings, model, faust_dataset_test, full_eval=True, visualize=True
        )

    # train
    # elif (path.exists(pretrained_path) and stage == 'auto') or stage == 'main':

    elif path.exists(trained_path) and stage == "eval_metrics":
        print("Evaluating on SHREC19 metrics... This would take ages..")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        faust_dataset_test = SHREC19Dataset(
            root=SHREC19_DIR, train=False, mode=settings.data_mode
        )
        eval_deform.test(
            settings, model, faust_dataset_test, full_eval=True, visualize=False
        )
    else:
        raise NotImplementedError()


def routine_smal_tosca(stage="auto", load_state=True):
    settings = training_configs()
    settings.dataset = "smaltosca"
    settings.input_params()
    # settings.identifier='s3knognn'

    target_shape = "horse"
    # target_shape = 'cat'
    # target_shape = 'centaur'
    # target_shape = 'david'

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    # init_model(trained_path, settings)

    # trained_path = get_model_name(False, settings,epoch='29')

    print("found pretrained:", path.exists(pretrained_path), pretrained_path)
    print("found trained:", path.exists(trained_path), trained_path)
    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":
        print("Evaluating on Tosca on class {}...".format(target_shape))
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        dataset_test = ToscaDataset(
            root=TOSCA_DIR,
            train=False,
            mode=settings.data_mode,
            target_shape=target_shape,
        )
        eval_deform.test(settings, model, dataset_test, full_eval=True, visualize=True)
    elif path.exists(trained_path) and stage == "eval_metrics":
        print(
            "Evaluating on Tosca on class {}. This would take ages...".format(
                target_shape
            )
        )
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        dataset_test = ToscaDataset(
            root=TOSCA_DIR,
            train=False,
            mode=settings.data_mode,
            target_shape=target_shape,
        )
        eval_deform.test(settings, model, dataset_test, full_eval=True, visualize=True)
    elif path.exists(trained_path) and stage == "eval_partial_metrics":
        print(
            "Evaluating on Tosca on class {}. This would take ages...".format(
                target_shape
            )
        )
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        dataset_test = ShrecPartialDataset(
            root=SHREC_PARTIAL_DIR, train=False, mode="cuts", target_shape=target_shape
        )
        eval_deform.test(settings, model, dataset_test, full_eval=True, visualize=True)
    else:
        print("Training on SMAL and test on Tosca with class {}".format(target_shape))
        dataset_trainval = SMALDataset(
            SMAL_DIR, train=True, mode=settings.data_mode, target_shape=target_shape
        )
        train_mask, val_mask = create_validation_masks(
            dataset_trainval, VALIDATION_SPLIT
        )
        dataset_test = ToscaDataset(
            root=TOSCA_DIR,
            train=False,
            mode=settings.data_mode,
            target_shape=target_shape,
        )

        val_loader = DataLoader(
            dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(dataset_trainval), val_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )
        loader_train = DataLoader(
            dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(dataset_trainval), train_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )
        feat_trainer = trainer(
            settings,
            loader_train,
            val_loader,
            dataset_test=dataset_test,
            eval_type=eval_deform.test,
            pretrain=False,
            model_path=trained_path,
        )
        feat_trainer.train()


def routine_body(stage="auto", load_state=True):
    settings = training_configs()
    settings.input_params()

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":

        print("Saving for FMNet...")

        root = "../../../dataset/MPI-FAUST/"
        mpi_dataset = FaustDataset(root, True)
        # test_loader = DataLoader(mpi_dataset, batch_size=32)
        model, _, _ = init_model(get_model_path(False, settings), settings)
        save_for_fmnet.test(settings, model, mpi_dataset)

        # for e in range(18,19):
        #     # model = init_model(get_model_name(False, settings,epoch=str(e)), settings)
        #     model = init_model(get_model_name(False, settings), settings)

        #     # test_nonrigid.test(settings, model, modelnet_dataset_test)

        #     test_match.test(settings, model, match3d_dataset_test)
        #     # test_match.test_visualize_desc(settings, model, match3d_dataset_test)

    # train
    elif (path.exists(pretrained_path) and stage == "auto") or stage == "main":

        print("Training on Body...")

        body_dataset_train = BodyDataset(
            root=BODY_DIR, train=True, mode=settings.data_mode
        )
        body_dataset_val = BodyDataset(
            root=BODY_DIR, train=False, mode=settings.data_mode
        )
        body_val_loader = DataLoader(body_dataset_val, batch_size=settings.mb_size)
        body_loader_train = DataLoader(body_dataset_train, batch_size=settings.mb_size)
        feat_trainer = trainer(
            settings,
            body_loader_train,
            body_val_loader,
            dataset_test=None,
            pretrain=False,
            model_path=pretrained_path,
        )
        feat_trainer.train()

        print("Saving for FMNet...")

        root = "../../../dataset/MPI-FAUST/"
        mpi_dataset = FaustDataset(root, True)
        # test_loader = DataLoader(mpi_dataset, batch_size=32)
        model, _, _ = init_model(get_model_path(False, settings), settings)
        save_for_fmnet.test(settings, model, mpi_dataset)


def routine_smal(stage="auto", load_state=True):
    settings = training_configs()
    settings.file_tag = "smal"

    pretrained_path = get_model_path(True, settings)
    trained_path = get_model_path(False, settings)

    # evaluate
    if (path.exists(trained_path) and stage == "auto") or stage == "eval":

        # print("Saving for FMNet...")

        # root='../../../dataset/MPI-FAUST/'
        # mpi_dataset=FaustDataset(root,True)
        # # test_loader = DataLoader(mpi_dataset, batch_size=32)
        # model = init_model(get_model_name(False, settings), settings)
        # save_for_fmnet.test(settings, model, mpi_dataset)

        print("Evaluating on SMAL...")
        model, _, _ = init_model(get_model_path(False, settings), settings)
        # faust_dataset_test= FaustscanDataset(root=MPIFAUST_DIR,train=False,mode=settings.data_mode,test_folder='test/')
        faust_dataset_test = SMALDataset(
            root=SMAL_DIR, train=False, mode=settings.data_mode
        )
        eval_deform.test(
            settings,
            model,
            faust_dataset_test,
        )

    # train
    elif (path.exists(pretrained_path) and stage == "auto") or stage == "main":

        print("Training on SMAL...")
        dataset_trainval = SMALDataset(
            root=SMAL_DIR, train=True, mode=settings.data_mode
        )
        train_mask, val_mask = create_validation_masks(
            dataset_trainval, VALIDATION_SPLIT
        )
        faust_dataset_test = SMALDataset(
            root=SMAL_DIR, train=True, sample_per_frame=1, mode=settings.data_mode
        )  # TODO: train should be false?
        # test_deform.test(settings,model,faust_dataset_test)
        faust_val_loader = DataLoader(
            dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(dataset_trainval), val_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )
        faust_loader_train = DataLoader(
            dataset_trainval,
            batch_sampler=BatchSampler(
                MaskedSampler(len(dataset_trainval), train_mask),
                batch_size=settings.mb_size,
                drop_last=False,
            ),
        )

        # model,_,_ = init_model(get_model_name(False, settings), settings)
        # test_deform.test(settings,model,faust_dataset_test)

        feat_trainer = trainer(
            settings,
            faust_loader_train,
            faust_val_loader,
            dataset_test=faust_dataset_test,
            eval_type=eval_deform.test,
            pretrain=False,
            model_path=trained_path,
        )
        feat_trainer.train()

        # print("Saving for FMNet...")

        # root = '../../../dataset/MPI-FAUST/'
        # mpi_dataset = FaustDataset(root, True)
        # # test_loader = DataLoader(mpi_dataset, batch_size=32)
        # model = init_model(get_model_name(False, settings), settings)
        # save_for_fmnet.test(settings, model, mpi_dataset)
