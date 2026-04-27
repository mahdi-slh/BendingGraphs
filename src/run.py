from utils.utils import *
from configs import *
from routines import *


if __name__ == "__main__":

    # routine_modelnet('main')
    # routine_body('main')
    # routine_body('eval')
    # routine_body('auto')
    # routine_smal('main')
    # routine_faust('auto')
    # routine_faust('eval')

    # routine_surreal_frust('eval_metrics')
    # routine_smal_tosca('eval')

    # routine_surreal_shrec19('eval')
    # routine_smal_tosca('eval_metrics')

    routine_modelnet_mesh("auto")
    # routine_smal_tosca('eval_metrics')

    # routine_faust('main')
    # routine_faust('eval')
    # routine_surreal_faust('eval')
    # routine_surreal_faust('eval')
    # routine_surreal_faust("eval_metrics")

    # settings = training_configs()
    # settings.input_params()

    # pretrained_path = get_model_name(True, settings)
    # trained_path = get_model_name(False, settings)

    # # match3d_dataset_test = Match3DDataset(MATCH3D_DIR)
    # # test_match.test(settings, model, match3d_dataset_test)

    # # evaluate
    # if path.exists(trained_path):

    #     print("Evaluation.")

    #     match3d_dataset_test = Match3DDataset(MATCH3D_DIR)
    #     # modelnet_dataset_test = modelnet_dataset(MODELNET_DIR, train=False,mode=settings.data_mode)
    #     for e in range(18,19):
    #         # model = init_model(get_model_name(False, settings,epoch=str(e)), settings)
    #         model = init_model(get_model_name(False, settings), settings)

    #         # test_rt.test(settings, model, modelnet_dataset_test)

    #         test_match.test(settings, model, match3d_dataset_test)
    #         # test_match.test_visualize_desc(settings, model, match3d_dataset_test)

    #     # evaluatemodelnet.evaluate_model(model, settings, modelnet_dataset_test, True)
    #     # fat_dataset_test = FATDataset (FAT_DIR)
    #     # evaluatefat.evaluate_model(model, settings, modelnet_dataset_test)

    # # train
    # elif path.exists(pretrained_path):

    #     print("Training 2.")

    #     match3d_dataset_test =Match3DDataset(MATCH3D_DIR)
    #     modelnet_dataset_train = ModelNetDataset(MODELNET_DIR, train=True,mode=settings.data_mode)
    #     modelnet_dataset_test = ModelNetDataset(MODELNET_DIR, train=False,mode=settings.data_mode)
    #     model = pretrain(settings, modelnet_dataset_train,modelnet_dataset_test,dataset_test_match=match3d_dataset_test, pretrain=False, model_path=pretrained_path)

    # # pretrain
    # else:

    #     print("Training 1.")

    #     synprim_sub_dirs = [1]

    #     # synprim_dataset = SynPrimDataset(root=SYN_PRIM_DIR, dirs=synprim_sub_dirs)

    #     body_dataset_train= BodyDataset(root=BODY_DIR,train=True,mode=settings.data_mode)
    #     body_dataset_val = BodyDataset(root=BODY_DIR,train=False,mode=settings.data_mode)

    #     modelnet_dataset_test = ModelNetDataset(MODELNET_DIR, train=False,mode=settings.data_mode)
    #     match3d_dataset_test =Match3DDataset(MATCH3D_DIR)
    #     # model = pretrain(settings, synprim_dataset,modelnet_dataset_test,dataset_test_match=match3d_dataset_test, pretrain=True)
    #     feat_trainer=trainer(settings, body_dataset_train,body_dataset_train,dataset_test_match=body_dataset_val, pretrain=True)
    #     # model = pretrain(settings, body_dataset_train,body_dataset_train,dataset_test_match=body_dataset_val, pretrain=True)
    #     feat_trainer.train()
    #     print("Training 2.")

    #     modelnet_dataset_train = ModelNetDataset(MODELNET_DIR, train=True,mode=settings.data_mode)

    #     model = pretrain(settings, modelnet_dataset_train,modelnet_dataset_test,dataset_test_match=match3d_dataset_test, pretrain=False, model_path=pretrained_path)
